from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
import os

app = Flask(__name__)

# Load the model (cached in the container's memory/disk)
embeddings = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-xl",
    cache_folder=os.getenv("HF_CACHE", "/cache/huggingface")
)
print("LLM Model loaded successfully!")

@app.route('/generate-embeddings', methods=['POST'])
def generate_embeddings():
    try:
        # Get text chunks from the request
        data = request.json
        texts = data.get("texts", [])
        
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        
        # Generate embeddings
        embeddings_list = embeddings.embed_documents(texts)
        return jsonify({"embeddings": embeddings_list})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)
