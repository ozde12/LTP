from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

def train_model():
    # 1. Load tokenizer & model
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    
    # 2. Load dataset
    #    Your CSV has two columns: "source-text", "target-text"
    #    We'll call everything "train" for now, then split in memory.
    dataset = load_dataset(
        "csv",
        data_files={"train": "data/processed/alicesegments.csv"}
    )
    # The dataset now has a single split: dataset["train"]
    
    # 3. Create a train/test split in memory
    train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_ds = train_test_split["train"]
    test_ds  = train_test_split["test"]

    # 4. Define a preprocessing function
    def preprocess_function(examples):
        # "source-text" is your paragraph or text input
        inputs = examples["source-text"]
        # "target-text" is your question (or answer, etc.)
        targets = examples["target-text"]
        
        # Tokenize the inputs
        model_inputs = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize the targets (labels)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=64,
                truncation=True,
                padding="max_length"
            )
        
        # Assign labels to model inputs
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs

    # 5. Apply tokenization to both train & test sets
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_test  = test_ds.map(preprocess_function, batched=True)

    # 6. Define training arguments
    training_args = TrainingArguments(
        output_dir="outputs/model_checkpoints",
        evaluation_strategy="epoch",  # Evaluate after each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=10_000,
        logging_dir="logs",
        fp16=True,  # Mixed precision (use if you have a GPU that supports it)
    )

    # 7. Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    # 8. Train the model
    trainer.train()

    # 9. Save final model
    trainer.save_model("outputs/final_model")

if __name__ == "__main__":
    train_model()
