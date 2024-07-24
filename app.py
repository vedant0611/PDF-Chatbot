import os
import pickle
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import faiss

app = FastAPI()

model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

class QueryRequest(BaseModel):
    query: str
    pdf_name: str

def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    if len(embeddings) == 0:
        raise ValueError("Embeddings are empty. Ensure that the chunks are properly created.")
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    return faiss_index

def query_faiss_index(index, query):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k=3)
    return indices[0]

def answer_question(docs, query):
    question_answering_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    context = " ".join(docs)
    answer = question_answering_pipeline(question=query, context=context)
    return answer['answer']

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_reader = PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in PDF.")

        store_name = file.filename[:-4]  # Remove the .pdf extension

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                faiss_index = pickle.load(f)
        else:
            try:
                faiss_index = create_faiss_index(chunks)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(faiss_index, f)
                with open(f"{store_name}.txt", "w") as f:
                    f.write(text)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        return {"filename": file.filename, "message": "PDF uploaded and processed successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query(request: QueryRequest):
    try:
        pdf_path = f"{request.pdf_name}.pkl"
        print(f"Checking for file: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=400, detail="PDF not found. Please upload the PDF first.")

        with open(pdf_path, "rb") as f:
            faiss_index = pickle.load(f)

        with open(f"{request.pdf_name}.txt", "r") as f:
            text = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        indices = query_faiss_index(faiss_index, request.query)
        docs = [chunks[i] for i in indices]
        response = answer_question(docs, request.query)
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
