from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

def train_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    
    # Load dataset
    dataset = datasets.load_dataset("data/processed/book_segments.csv")
    
    # Tokenize dataset
    def preprocess_function(examples):
        inputs = examples['segments']
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        return model_inputs

    train_test_split = dataset['train'].train_test_split(test_size=0.2)
    tokenized_datasets = train_test_split.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir="outputs/model_checkpoints",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=10_000,
        logging_dir="logs",
        fp16=True,  # Mixed precision
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
    )

    trainer.train()
    trainer.save_model("outputs/final_model")

if __name__ == "__main__":
    train_model()
