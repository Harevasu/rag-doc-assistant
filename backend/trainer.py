"""
Trainer Module.
Fine-tunes a language model on document data for question answering.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

from config import DATA_DIR, CHUNK_SIZE
from vector_store import get_all_documents, get_collection


# Training configuration
MODEL_NAME = "google/flan-t5-base"
TRAINED_MODEL_DIR = DATA_DIR / "trained_model"
TRAINING_DATA_PATH = DATA_DIR / "training_data.json"


def get_all_chunks() -> List[Dict[str, Any]]:
    """Get all document chunks from the vector store."""
    collection = get_collection()
    if collection is None:
        return []
    
    results = collection.get(include=["documents", "metadatas"])
    
    chunks = []
    if results and results.get("documents"):
        for i, doc in enumerate(results["documents"]):
            metadata = results["metadatas"][i] if results.get("metadatas") else {}
            chunks.append({
                "text": doc,
                "metadata": metadata
            })
    
    return chunks


def generate_qa_pairs(chunks: List[Dict[str, Any]], pairs_per_chunk: int = 2) -> List[Dict[str, str]]:
    """
    Generate question-answer pairs from document chunks.
    Uses simple template-based generation.
    
    Args:
        chunks: List of document chunks
        pairs_per_chunk: Number of Q&A pairs to generate per chunk
        
    Returns:
        List of {"question": ..., "answer": ..., "context": ...} dicts
    """
    qa_pairs = []
    
    question_templates = [
        "What does this section say about {topic}?",
        "Can you explain {topic}?",
        "What is mentioned about {topic}?",
        "Describe {topic} based on the document.",
        "What information is provided about {topic}?",
    ]
    
    for chunk in chunks:
        text = chunk["text"]
        filename = chunk["metadata"].get("filename", "document")
        
        # Extract key phrases (simple approach: use sentences)
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        
        if not sentences:
            continue
        
        # Generate Q&A pairs
        for _ in range(min(pairs_per_chunk, len(sentences))):
            sentence = random.choice(sentences)
            
            # Extract a topic from the sentence (first few words)
            words = sentence.split()
            if len(words) < 5:
                continue
            
            topic = " ".join(words[:min(5, len(words))])
            
            # Create Q&A pair
            template = random.choice(question_templates)
            question = template.format(topic=topic.lower())
            
            qa_pairs.append({
                "question": question,
                "answer": sentence,
                "context": text,
                "source": filename
            })
    
    return qa_pairs


def prepare_training_data() -> str:
    """
    Prepare training data from all indexed documents.
    
    Returns:
        Path to the training data file
    """
    print("📊 Gathering document chunks...")
    chunks = get_all_chunks()
    
    if not chunks:
        raise ValueError("No documents indexed. Please upload documents first.")
    
    print(f"📊 Found {len(chunks)} chunks")
    print("📊 Generating Q&A pairs...")
    
    qa_pairs = generate_qa_pairs(chunks, pairs_per_chunk=3)
    
    print(f"📊 Generated {len(qa_pairs)} Q&A pairs")
    
    # Save training data
    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Training data saved to {TRAINING_DATA_PATH}")
    return str(TRAINING_DATA_PATH)


def train_model(
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5
) -> str:
    """
    Fine-tune the model on prepared training data.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        
    Returns:
        Path to the trained model
    """
    # Import here to avoid slow startup
    from transformers import (
        T5ForConditionalGeneration,
        T5Tokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForSeq2Seq
    )
    from datasets import Dataset
    import torch
    
    print("🚀 Loading base model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Load training data
    if not TRAINING_DATA_PATH.exists():
        prepare_training_data()
    
    with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    print(f"🚀 Loaded {len(qa_pairs)} training examples")
    
    # Prepare dataset
    def preprocess(examples):
        inputs = [f"question: {ex['question']} context: {ex['context']}" for ex in examples]
        targets = [ex['answer'] for ex in examples]
        
        model_inputs = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Create dataset
    dataset = Dataset.from_list(qa_pairs)
    
    # Tokenize
    print("🚀 Tokenizing dataset...")
    tokenized = dataset.map(
        lambda x: preprocess([x]),
        remove_columns=dataset.column_names
    )
    
    # Split into train/eval
    split = tokenized.train_test_split(test_size=0.1)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(TRAINED_MODEL_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=str(DATA_DIR / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("🚀 Starting training...")
    trainer.train()
    
    # Save the model
    print("🚀 Saving trained model...")
    trainer.save_model(str(TRAINED_MODEL_DIR))
    tokenizer.save_pretrained(str(TRAINED_MODEL_DIR))
    
    print(f"✅ Model saved to {TRAINED_MODEL_DIR}")
    return str(TRAINED_MODEL_DIR)


class FineTunedModel:
    """Wrapper for the fine-tuned model for inference."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self):
        """Load the trained model."""
        if self._loaded:
            return
        
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        import torch
        
        if not TRAINED_MODEL_DIR.exists():
            raise ValueError(
                f"No trained model found at {TRAINED_MODEL_DIR}. "
                "Please train the model first using the /train endpoint."
            )
        
        print("📥 Loading fine-tuned model...")
        self.tokenizer = T5Tokenizer.from_pretrained(str(TRAINED_MODEL_DIR))
        self.model = T5ForConditionalGeneration.from_pretrained(str(TRAINED_MODEL_DIR))
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("📥 Model loaded on GPU")
        else:
            print("📥 Model loaded on CPU")
        
        self._loaded = True
    
    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer using the fine-tuned model.
        
        Args:
            question: User's question
            context: Document context
            
        Returns:
            Generated answer
        """
        if not self._loaded:
            self.load()
        
        import torch
        
        # Prepare input
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


# Global instance
fine_tuned_model = FineTunedModel()


def is_model_trained() -> bool:
    """Check if a trained model exists."""
    return TRAINED_MODEL_DIR.exists() and (TRAINED_MODEL_DIR / "config.json").exists()
