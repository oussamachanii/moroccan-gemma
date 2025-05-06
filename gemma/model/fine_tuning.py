import os
import torch
from typing import Dict, Any
from datasets import load_dataset, Dataset
from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from .base import GemmaModel

def prepare_dataset(dataset_path: str):
    """
    Prepare the Moroccan dialect dataset for fine-tuning.
    
    Args:
        dataset_path: Path to the processed dataset directory
        
    Returns:
        A HuggingFace dataset object
    """
    try:
        # Try to load from Hugging Face
        dataset = load_dataset("json", data_files=os.path.join(dataset_path, "moroccan_dialects.json"))
    except Exception as e:
        print(f"Error loading dataset from path: {e}")
        # Fallback to loading from local file
        if os.path.exists(os.path.join(dataset_path, "moroccan_dialects.json")):
            with open(os.path.join(dataset_path, "moroccan_dialects.json"), "r", encoding="utf-8") as f:
                import json
                data = json.load(f)
                
            # Create a dataset
            dataset = Dataset.from_dict({
                "text": [item["text"] for item in data],
                "translation": [item.get("translation", "") for item in data]
            })
        else:
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
    return dataset

def fine_tune_model(
    model: GemmaModel,
    dataset_path: str,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    output_dir: str = "/gemma/fine_tuned"
) -> Dict[str, Any]:
    """
    Fine-tune the Gemma model on Moroccan dialect data.
    
    Args:
        model: The GemmaModel instance
        dataset_path: Path to the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        output_dir: Directory to save the fine-tuned model
        
    Returns:
        Training metrics
    """
    print(f"Preparing to fine-tune model on Moroccan dialect data from {dataset_path}")
    
    # Prepare dataset
    dataset = prepare_dataset(dataset_path)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return model.tokenizer(
            examples["text"], 
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=model.tokenizer,
        mlm=False  # We're doing causal language modeling
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
    )
    
    # Trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting fine-tuning...")
    train_result = trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model(output_dir)
    model.tokenizer.save_pretrained(output_dir)
    
    print(f"Fine-tuning completed. Model saved to {output_dir}")
    
    return {
        "train_loss": train_result.training_loss,
        "epochs": epochs,
        "model_path": output_dir
    }