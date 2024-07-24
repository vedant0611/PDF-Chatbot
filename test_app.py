import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_upload_pdf():
    with open("Vedant-Tadla-Resume.pdf", "rb") as pdf:
        response = client.post("/upload_pdf/", files={"file": pdf})
    assert response.status_code == 200
    assert response.json()["message"] == "PDF uploaded and processed successfully."

def test_query():
    response = client.post("/query/", json={"query": "What is the document about?", "pdf_name": "Vedant-Tadla-Resume"})
    assert response.status_code == 200
    assert "response" in response.json()
