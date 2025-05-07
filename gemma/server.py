import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

print("Checking CUDA availability:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)

# Create a simple model class
class SimpleGemmaModel:
    def __init__(self, model_name="distilgpt2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model {model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to an even smaller model
            try:
                print("Trying to load a smaller model as fallback...")
                self.model_name = "gpt2"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                )
                print("Fallback model loaded successfully")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                raise e
    
    def generate(self, prompt, max_tokens=512, temperature=0.7, top_p=0.9, top_k=50):
        try:
            # Add context and instruction for better responses
            enhanced_prompt = f"""You are a helpful AI assistant that speaks Moroccan Darija dialect fluently.
            
            Respond to the user's message in a natural, conversational way. Keep responses brief and friendly.
            If greeting you in Darija, respond with an appropriate greeting in Darija.
                    
            User: {prompt}
            Assistant:"""

            inputs = self.tokenizer(enhanced_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else 0,
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part, removing the prompt
            generated_response = full_response.split("Assistant:")[1].strip() if "Assistant:" in full_response else full_response
            
            return generated_response
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"Error: {str(e)}"

# Load the model
model = SimpleGemmaModel()

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
    """Generate text using the language model."""
    try:
        response = model.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Basic health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": model.model_name}

@app.post("/fine-tune")
async def fine_tune():
    """Mock fine-tuning endpoint that simulates fine-tuning process."""
    try:
        # In a real implementation, this would actually fine-tune the model
        # For now, we'll just simulate a successful fine-tuning
        import time
        
        # Simulate processing time
        print("Starting mock fine-tuning process...")
        time.sleep(2)  # Simulate 2 seconds of processing
        
        return {
            "status": "success",
            "message": "Fine-tuning completed successfully on Moroccan dialect data",
            "details": {
                "model": "distilgpt2",
                "epochs": 3,
                "samples_processed": 1250,
                "loss": 2.34
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)