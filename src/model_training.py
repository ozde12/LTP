from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

def train_model():
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    
    # Load dataset
    dataset = load_dataset("csv", data_files="data/processed/alice_segments.csv")
    dataset = dataset["train"].train_test_split(test_size=0.2)
    
    # Tokenize dataset
    def preprocess_function(examples):
        inputs = examples['segments']
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="outputs/model_checkpoints",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=10_000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
    )

    trainer.train()

if __name__ == "__main__":
    train_model()
