import os
import requests
from io import BytesIO
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

app = Flask(__name__)

# Replace this with the LLM server's URL
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://llm-server:6000")

# Get PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ''
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Send chunks to LLM server for embeddings
    response = requests.post(f"{LLM_SERVER_URL}/generate-embeddings", json={"texts": text_chunks})
    response.raise_for_status()  # Raise an error if the LLM server fails

    # Extract embeddings from the response
    embeddings = response.json()["embeddings"]

    # Build FAISS vector store
    vector_store = FAISS.from_embeddings(text_chunks, embeddings)
    return vector_store

@app.route('/', methods=["GET"])
def root_route():
    return "Welcome to Langchain RAG Pipeline"

@app.route('/process-pdf', methods=["POST"])
def process_pdf():
    try:
        pdf_files = request.files.getlist('selectedFiles')
        if not pdf_files:
            return jsonify({"error": "No files received"})
        
        pdf_docs = [BytesIO(file.read()) for file in pdf_files]
        pdf_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(pdf_text)
        
        vectorstore = get_vector_store(text_chunks)
        document_ids = list(vectorstore.docstore._dict.keys())
        return jsonify({
            "message": "PDF processed successfully",
            "chunk_count": len(text_chunks),
            "vector_store_status": "Vector store updated with new text",
            "document_ids": document_ids
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
