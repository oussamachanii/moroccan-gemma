import os
from flask import Flask, request, jsonify, render_template
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
GEMMA_API_URL = os.getenv("GEMMA_API_URL", "http://gemma:8080")
CHROMA_API_URL = os.getenv("CHROMA_API_URL", "http://chroma:8000")

@app.route('/')
def index():
    """Render the main interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chat interactions."""
    data = request.json
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Query ChromaDB for relevant context
    context_data = get_context_from_chroma(user_input)
    
    # Generate response using Gemma 3 model with context
    response = query_gemma_model(user_input, context_data)
    
    return jsonify({"response": response})

def get_context_from_chroma(query):
    """Retrieve relevant context from ChromaDB."""
    try:
        response = requests.post(
            f"{CHROMA_API_URL}/api/v1/collections/moroccan_dialect/query",
            json={
                "query_texts": [query],
                "n_results": 5
            }
        )
        response.raise_for_status()
        results = response.json()
        
        # Extract documents from results
        if 'documents' in results and results['documents']:
            return results['documents'][0]  # Return only the most relevant documents
        return []
    except Exception as e:
        print(f"Error retrieving context from ChromaDB: {e}")
        return []

def query_gemma_model(query, context):
    """Query the Gemma 3 model with the user input and context."""
    try:
        # Combine context with user query
        context_str = " ".join(context) if context else ""
        prompt = f"Context: {context_str}\n\nUser question in Moroccan dialect: {query}\n\nPlease respond in Moroccan dialect (Darija):"
        
        response = requests.post(
            f"{GEMMA_API_URL}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.7
            }
        )
        response.raise_for_status()
        result = response.json()
        
        return result.get('response', "Sorry, I couldn't generate a response.")
    except Exception as e:
        print(f"Error querying Gemma model: {e}")
        return f"Error: {str(e)}"

@app.route('/api/fine-tune', methods=['POST'])
def fine_tune():
    """API endpoint to trigger fine-tuning."""
    try:
        response = requests.post(f"{GEMMA_API_URL}/fine-tune")
        response.raise_for_status()
        return jsonify({"status": "success", "message": "Fine-tuning process started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)