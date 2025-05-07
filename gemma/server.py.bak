import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from model.base import GemmaModel
from model.fine_tuning import fine_tune_model

app = FastAPI()

# Load Gemma model
gemma_model = GemmaModel()

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
    """Generate text using the Gemma 3 model."""
    try:
        response = gemma_model.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FineTuneRequest(BaseModel):
    dataset_path: Optional[str] = "/data/processed"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5

@app.post("/fine-tune")
async def finetune(request: FineTuneRequest):
    """Trigger fine-tuning of the Gemma model on Moroccan dialect data."""
    try:
        result = fine_tune_model(
            model=gemma_model,
            dataset_path=request.dataset_path,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate
        )
        return {"status": "success", "message": f"Fine-tuning completed: {result}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)