import os
import requests
import json
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Configuration
GEMMA_API_URL = os.getenv("GEMMA_API_URL", "http://gemma:8080")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

class GenerateResponse(BaseModel):
    response: str

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text using the Gemma 3 model via Docker Model Runner."""
    try:
        # Format request for Docker Model Runner OpenAI-compatible API
        payload = {
            "messages": [
                {"role": "user", "content": request.prompt}
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "top_k": request.top_k
        }
        
        # Send request to Docker Model Runner
        response = requests.post(
            f"{GEMMA_API_URL}/v1/chat/completions", 
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"Error from Gemma API: {response.text}")
            raise HTTPException(status_code=response.status_code, 
                                detail=f"Gemma API error: {response.text}")
        
        # Parse response
        response_data = response.json()
        generated_text = response_data["choices"][0]["message"]["content"]
        
        return {"response": generated_text}
    except Exception as e:
        print(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("adapter:app", host="0.0.0.0", port=8080, reload=True)