import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional

class GemmaModel:
    """Wrapper for the Gemma 3 model."""
    
    def __init__(self, model_name: str = "google/gemma-3-1b"):
        """
        Initialize the Gemma model.
        
        Args:
            model_name: The name of the model to load from HuggingFace
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Gemma model {model_name} on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        print(f"Gemma model loaded successfully.")
    
    def generate(self, 
                prompt: str, 
                max_tokens: int = 512, 
                temperature: float = 0.7,
                top_p: float = 0.9,
                top_k: int = 50) -> str:
        """
        Generate text using the Gemma model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and clean the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part, removing the prompt
        response = response[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        
        return response.strip()
    
    def save_model(self, output_dir: str):
        """Save the model and tokenizer to the specified directory."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str):
        """Load a model from a local directory."""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"Model loaded from {model_dir}")