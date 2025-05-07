import os
from flask import Flask, request, jsonify, render_template
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
GEMMA_API_URL = os.getenv("GEMMA_API_URL", "http://gemma-adapter:8080")
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
    """Simulate retrieving relevant context."""
    # Since we're not actually connecting to ChromaDB's API,
    # we'll just return a simple mock response for now
    try:
        # In a real setup, you would connect to ChromaDB directly
        # But for now, let's just return a default context
        print(f"Searching context for: {query}")
        return ["Context for Moroccan dialect: Darija is the primary spoken language in Morocco."]
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []
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
    """Query the Gemma model with the user input and context."""
    try:
        # Use a simpler prompt format without repetitive instructions
        prompt = f"""You are a helpful assistant fluent in Moroccan Darija.

User: {query}

Assistant (respond in Darija, be brief and natural):"""
        
        # Set a smaller max_tokens to prevent long repetitive responses
        response = requests.post(f"{GEMMA_API_URL}/generate", 
                              json={"prompt": prompt, "max_tokens": 100, "temperature": 0.7})
        response.raise_for_status()
        result = response.json()
        
        # Get the generated text
        generated_text = result.get('response', "")
        
        # Provide fallback responses for common phrases if the model fails
        if not generated_text or "Please respond in" in generated_text or "I am the first person" in generated_text:
            if "labas" in query.lower() or "lbas" in query.lower() or "3lik" in query.lower():
                return "Hamdullah, labas. Nta/Nti labas?"
            elif "salam" in query.lower() or "hello" in query.lower() or "hi" in query.lower():
                return "Salam! Kifash nqder n3awnek lyoum?"
            elif "smiya" in query.lower() or "smitk" in query.lower() or "name" in query.lower():
                return "Ana Gemma, l'assistant dial Darija. Ntuma smiytkum?"
            else:
                return "Smehli, ma fhemteksh mezyan. Momkin t3awed s'question dyalek?"
        
        return generated_text
    except Exception as e:
        print(f"Error querying Gemma model: {e}")
        return f"Error: {str(e)}"

@app.route('/api/fine-tune', methods=['POST'])
def fine_tune():
    """API endpoint to trigger fine-tuning."""
    try:
        # Add debugging information
        print(f"Attempting to connect to Gemma API at {GEMMA_API_URL}/fine-tune")
        
        # Set a longer timeout
        response = requests.post(f"{GEMMA_API_URL}/fine-tune", timeout=10)
        
        try:
            response.raise_for_status()
            return jsonify({"status": "success", "message": "Fine-tuning process started"})
        except Exception as e:
            print(f"Error from Gemma API: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    except Exception as e:
        print(f"Connection error: {e}")
        return jsonify({
            "status": "error", 
            "message": str(e),
            "details": "The adapter service might not be running properly. Check your Docker logs."
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)