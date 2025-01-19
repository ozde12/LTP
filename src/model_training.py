import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

def main():
    df = pd.read_csv("data/processed/alicesegments.csv")
    # Ensure it has columns "source-text" and "target-text"

    # The CSV columns are "source-text" (prompt context) and "target-text" (desired question).
    # We'll rename them to something consistent if you like:
    dataset = Dataset.from_pandas(df)

    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = dataset["train"]
    eval_dataset  = dataset["test"]

    model_name = "t5-small"  # You can also try "t5-base", "google/flan-t5-small", etc.
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    max_source_length = 512
    max_target_length = 128

    def preprocess_function(examples):
        # tokenize the input (the text from "source-text")
        model_inputs = tokenizer(
            examples["source-text"], 
            max_length=max_source_length, 
            truncation=True
        )
        # tokenize the target (the text from "target-text")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target-text"], 
                max_length=max_target_length, 
                truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # map preprocessing onto the entire dataset
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval  = eval_dataset.map(preprocess_function,  batched=True)


    # data collator will dynamically pad inputs and labels to the
    # longest sequence in the batch, so no need to pad them all to max_length.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    training_args = TrainingArguments(
        output_dir="./trained_model",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator
    )

    trainer.train()

    # save the model
    trainer.save_model("./trained_model")
    tokenizer.save_pretrained("./trained_model")

    print("Training complete! Model saved to './trained_model'")

if __name__ == "__main__":
    main()
