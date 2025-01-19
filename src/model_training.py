from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

def train_model():
    # 1. Load tokenizer & model
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    
    # 2. Load dataset
    dataset = load_dataset(
        "csv",
        data_files={"train": "data/processed/alicesegments.csv"}
    )
    
    # 3. Create a train/test split
    train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_ds = train_test_split["train"]
    test_ds = train_test_split["test"]

    # 4. Define preprocessing function
    def preprocess_function(examples):
        inputs = examples["source-text"]
        targets = examples["target-text"]

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():  # Ensures proper target tokenization
            labels = tokenizer(
                targets,
                max_length=64,
                truncation=True,
                padding="max_length"
            )
        
        # Assign labels to model inputs
        model_inputs["labels"] = labels["input_ids"]
        
        # Debug: Print tokenization results
        print(f"Input Text: {inputs}")
        print(f"Tokenized Input IDs: {model_inputs['input_ids']}")
        print(f"Target Text: {targets}")
        print(f"Tokenized Target IDs: {labels['input_ids']}")
        print("-" * 80)

        # Debug: Check if labels contain all padding
        if all(label == tokenizer.pad_token_id for label in labels["input_ids"]):
            print(f"Warning: Label contains only padding for input: {inputs}")

        return model_inputs

    # 5. Tokenize datasets
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_test = test_ds.map(preprocess_function, batched=True)

    # 6. Define training arguments
    training_args = TrainingArguments(
        output_dir="outputs/model_checkpoints",
        evaluation_strategy="steps",
        eval_steps=500,
        logging_dir="logs",
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=1e-5,  # Reduced learning rate for stability
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch size
        gradient_clipping=1.0,  # Prevent exploding gradients
        label_smoothing_factor=0.1,  # Prevent overconfidence
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none"  # Disable external logging for simplicity
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test
    )

    # 8. Train model
    trainer.train()

    # 9. Save final model and tokenizer
    trainer.save_model("outputs/final_model")
    tokenizer.save_pretrained("outputs/final_model")

if __name__ == "__main__":
    train_model()
