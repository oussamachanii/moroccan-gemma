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

@app.get("/health")
async def health_check():
    try:
        response = requests.get(f"{GEMMA_API_URL}/health")
        if response.status_code == 200:
            return {"status": "healthy", "gemma_status": response.json()}
        return {"status": "unhealthy", "reason": f"Gemma API returned {response.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Forward request to Gemma model."""
    try:
        # Simple forwarding of the request
        payload = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k
        }
        
        # Send request to Gemma service
        response = requests.post(
            f"{GEMMA_API_URL}/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"Error from Gemma API: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Gemma API error: {response.text}")
        
        # Parse response
        response_data = response.json()
        return {"response": response_data["response"]}
    except Exception as e:
        print(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fine-tune")
async def fine_tune():
    """Forward fine-tuning request to the Gemma model."""
    try:
        # Send request to Gemma service
        response = requests.post(
            f"{GEMMA_API_URL}/fine-tune",
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"Error from Gemma API: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Gemma API error: {response.text}")
        
        # Return the response from Gemma
        return response.json()
    except Exception as e:
        print(f"Error initiating fine-tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("adapter:app", host="0.0.0.0", port=8080, reload=False)