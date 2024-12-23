import os
from io import BytesIO
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)

# Load HuggingFace model with cache folder environment variable
embeddings = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-xl",
    cache_folder=os.getenv("HF_CACHE", "./cache/huggingface")
)
print("Model loaded successfully!")

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
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

@app.route('/', methods=["GET"])
def root_route():
    return "Welcome to Langchain PDF Model"

@app.route('/process-pdf', methods=["POST"])
def process_pdf():
    try:
        pdf_files = request.files.getlist('selectedFiles')
        print("pdf_files:", pdf_files) 
        
        if not pdf_files:
            return jsonify({"error": "No files received"})
        
        pdf_docs = [BytesIO(file.read()) for file in pdf_files]
        
        pdf_text = get_pdf_text(pdf_docs)
        
        text_chunks = get_text_chunks(pdf_text)
        
        vectorstore = get_vector_store(text_chunks)
        document_ids = list(vectorstore.docstore._dict.keys())
        print("Document IDs:", document_ids)
        return jsonify({
            "message": "PDF processed successfully",
            "chunk_count": len(text_chunks),
            "vector_store_status": "Vector store updated with new text",
            "document_ids": document_ids
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    os.environ["HF_HOME"] = "./cache/huggingface"
    os.environ["HF_HUB_OFFLINE"] = "1"
    load_dotenv()
    app.run(host='0.0.0.0', port=5000, debug=True)
