# PDF Chat Application

## Overview

The PDF Chat Application allows users to upload PDF documents and interact with their content through a conversational interface. Built using FastAPI for the backend and Hugging Face models for natural language processing, this application enables users to extract and query information from PDFs.

## Features

- **Upload PDF:** Upload PDF files to be processed and stored.
- **Query PDF Content:** Ask questions about the uploaded PDF and receive answers based on the document’s content.

## Technologies Used

- **FastAPI:** For creating the API endpoints.
- **Hugging Face Transformers:** For question-answering capabilities. We use the DistilBERT model from Hugging Face, which is fine-tuned for question-answering tasks.
- **Sentence Transformers:** For generating embeddings of text chunks.
- **FAISS:** For efficient similarity search in the embeddings.
- **PyPDF2:** For reading and extracting text from PDF files.
- **Streamlit (Optional):** For building a user-friendly frontend interface.
- **pytest:** For testing the application.

## About Hugging Face

Hugging Face is an AI company specializing in Natural Language Processing (NLP). Their Transformers library provides state-of-the-art models for various NLP tasks including text generation, translation, and question-answering.

In this project, we use Hugging Face's `transformers` library to leverage the DistilBERT model. This model is a smaller, faster variant of BERT (Bidirectional Encoder Representations from Transformers) optimized for question-answering tasks. It helps in understanding and generating answers to questions based on the content of the uploaded PDF.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Pip (Python package manager)

### Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. Start the FastAPI server:

    ```bash
    uvicorn app:app --reload
    ```

   This will start the server on `http://127.0.0.1:8000`.

2. Open your browser and navigate to `http://127.0.0.1:8000/docs` to access the API documentation provided by FastAPI.

### API Endpoints

#### 1. Upload PDF

- **Endpoint:** `/upload_pdf/`
- **Method:** `POST`
- **Description:** Upload a PDF file for processing.
- **Request:**
  - **Body:** Multipart form-data with the file key (e.g., `file`)
- **Response:**
  - **Status Code:** 200
  - **Body:**
    ```json
    {
      "filename": "your_file_name.pdf",
      "message": "PDF uploaded and processed successfully."
    }
    ```

#### 2. Query PDF Content

- **Endpoint:** `/query/`
- **Method:** `POST`
- **Description:** Query the content of the uploaded PDF.
- **Request:**
  - **Body:** JSON object with `query` and `pdf_name` keys
    ```json
    {
      "query": "What is the document about?",
      "pdf_name": "your_file_name"
    }
    ```
- **Response:**
  - **Status Code:** 200
  - **Body:**
    ```json
    {
      "response": "The answer to your query."
    }
    ```

## Testing

1. To run the tests, use the following command:

    ```bash
    pytest
    ```

   Ensure you have a file named `test.pdf` in your project directory for testing purposes.

