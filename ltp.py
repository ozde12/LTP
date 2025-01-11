# Import necessary libraries
import os
import re
import nltk
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    TrainingArguments
)
from rouge_score import rouge_scorer

# Initialize constants

BOOK_FILE = r"C:\Users\ozdep\Documents\ltp final project\LTP\alice.txt"
MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = "./results"

# Step 1: Preprocessing the Text
def preprocess_text(file_path):
    """
    Read and clean text data.
    Segments the text into paragraphs for model input.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Remove Gutenberg license text
    start_idx = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end_idx = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    text = text[start_idx:end_idx] if start_idx != -1 and end_idx != -1 else text

    # Split into paragraphs
    paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
    return paragraphs

# Step 2: Prepare Input-Output Pairs
def create_input_output_pairs(paragraphs):
    """
    Generate input-output pairs for fine-tuning.
    Input: Narrative text.
    Output: Manually curated subjective questions (for initial fine-tuning).
    """
    examples = []
    for para in paragraphs:
        # Example: Replace this with actual subjective questions
        question = f"What do you think about the events in this paragraph: '{para[:50]}...'?"
        examples.append({"text": para, "question": question})
    return examples

# Step 3: Tokenization
def tokenize_data(examples, tokenizer, max_input_length=512, max_output_length=128):
    """
    Tokenize input and output texts.
    """
    inputs = tokenizer(
        examples["text"], max_length=max_input_length, truncation=True, padding="max_length"
    )
    outputs = tokenizer(
        examples["question"], max_length=max_output_length, truncation=True, padding="max_length"
    )
    return {"input_ids": inputs["input_ids"], "labels": outputs["input_ids"]}

# Step 4: Model Fine-Tuning
def fine_tune_model(dataset, tokenizer):
    """
    Fine-tune the model on the prepared dataset.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=1000,
        save_total_limit=2,
        logging_dir="./logs",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    return model

# Step 5: Evaluation Metrics
def evaluate_model(model, tokenizer, dataset):
    """
    Evaluate the fine-tuned model using metrics like ROUGE and BLEU.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    for example in dataset:
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs)
        generated_question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated Question: {generated_question}")
        print(f"Reference Question: {example['question']}")
        print(scorer.score(example["question"], generated_question))

# Main Execution
if __name__ == "__main__":
    # Preprocess text data
    paragraphs = preprocess_text(BOOK_FILE)

    # Create input-output pairs
    examples = create_input_output_pairs(paragraphs)
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict({"text": [e["text"] for e in examples], "question": [e["question"] for e in examples]})
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize dataset
    tokenized_dataset = dataset.map(lambda e: tokenize_data(e, tokenizer), batched=True)

    # Fine-tune model
    fine_tuned_model = fine_tune_model(tokenized_dataset, tokenizer)

    # Evaluate model
    evaluate_model(fine_tuned_model, tokenizer, examples)
