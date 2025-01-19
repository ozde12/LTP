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
    
    # 3. Create a train/test split in memory
    train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_ds = train_test_split["train"]
    test_ds  = train_test_split["test"]

    # 4. Define a preprocessing function
    def preprocess_function(examples):
        inputs = examples["source-text"]
        targets = examples["target-text"]
        
        # Tokenize the inputs
        model_inputs = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize the targets (labels) using text_target
        labels = tokenizer(
            text_target=targets,
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
        evaluation_strategy="steps",  # Evaluate every few steps
        eval_steps=500,  # Perform evaluation every 500 steps
        logging_dir="logs",
        logging_steps=100,  # Log every 100 steps
        save_steps=1000,  # Save the model every 1000 steps
        save_total_limit=2,  # Keep only the last 2 checkpoints
        learning_rate=5e-5,  # Increased learning rate
        per_device_train_batch_size=16,  # Use a larger batch size if memory allows
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,  # Mixed precision for faster training on compatible GPUs
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_loss",  # Use evaluation loss to track the best model
    )

    # 7. Create a Trainer instance with additional logging callback
    from transformers import TrainerCallback

    class LoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                print(f"Step {state.global_step}: {logs}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        callbacks=[LoggingCallback()]
    )

    # 8. Train the model
    trainer.train()

    # 9. Save final model and tokenizer
    trainer.save_model("outputs/final_model")
    tokenizer.save_pretrained("outputs/final_model")

if __name__ == "__main__":
    train_model()
