import os
import torch
from typing import List, Dict, Any, Optional
from .base import GemmaModel

class GemmaInference:
    """Inference class for the Gemma model"""
    
    def __init__(self, model: Optional[GemmaModel] = None, model_path: Optional[str] = None):
        """
        Initialize the inference engine.
        
        Args:
            model: Existing GemmaModel instance or None to create a new one
            model_path: Path to a fine-tuned model (if not using the base model)
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = GemmaModel()
            self.model.load_model(model_path)
        else:
            self.model = GemmaModel()
    
    def generate_response(self, 
                          query: str, 
                          context: Optional[List[str]] = None,
                          max_tokens: int = 512,
                          temperature: float = 0.7) -> str:
        """
        Generate a response to a query with optional context.
        
        Args:
            query: User query text
            context: Optional list of context documents
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            
        Returns:
            Generated response text
        """
        # Prepare the prompt with context if provided
        context_str = "\n".join(context) if context else ""
        
        if context_str:
            prompt = f"Context: {context_str}\n\nUser: {query}\n\nAssistant:"
        else:
            prompt = f"User: {query}\n\nAssistant:"
        
        # Generate the response
        response = self.model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response
    
    def answer_question(self, 
                        question: str, 
                        context: List[str], 
                        respond_in_darija: bool = True) -> str:
        """
        Answer a specific question using relevant context.
        
        Args:
            question: The question to answer
            context: List of context documents
            respond_in_darija: Whether to respond in Moroccan dialect
            
        Returns:
            Answer text
        """
        # Combine context
        context_text = "\n".join(context) if context else ""
        
        # Prepare prompt based on language preference
        if respond_in_darija:
            prompt = f"Context: {context_text}\n\nQuestion in Moroccan dialect: {question}\n\nPlease respond in Moroccan dialect (Darija):\n"
        else:
            prompt = f"Context: {context_text}\n\nQuestion in Moroccan dialect: {question}\n\nPlease respond in English:\n"
        
        # Generate answer
        answer = self.model.generate(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7
        )
        
        return answer
    
    def translate(self, text: str, target_language: str = "english") -> str:
        """
        Translate text between Moroccan dialect and another language.
        
        Args:
            text: Text to translate
            target_language: Target language (e.g., 'english')
            
        Returns:
            Translated text
        """
        # Detect source language and set up prompt
        if target_language.lower() == "english":
            prompt = f"Translate this Moroccan dialect text to English:\n\n{text}\n\nEnglish translation:"
        else:
            prompt = f"Translate this English text to Moroccan dialect (Darija):\n\n{text}\n\nDarija translation:"
        
        # Generate translation
        translation = self.model.generate(
            prompt=prompt,
            max_tokens=512,
            temperature=0.3  # Lower temperature for more precise translation
        )
        
        return translation