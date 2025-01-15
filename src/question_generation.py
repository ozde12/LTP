from transformers import T5Tokenizer, T5ForConditionalGeneration

def generate_subjective_questions(input_texts, model_path="outputs/model_checkpoints"):
    """
    Generates subjective questions from input texts using the fine-tuned model.
    
    Args:
        input_texts (list): A list of input text segments.
        model_path (str): Path to the fine-tuned model.

    Returns:
        list: Generated subjective questions.
    """
    # Load tokenizer and fine-tuned model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    # Tokenize input texts
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate questions
    outputs = model.generate(inputs.input_ids, max_length=64, num_beams=5, early_stopping=True)

    # Decode generated questions
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

if __name__ == "__main__":
    # Example input texts
    input_texts = [
        "The Gryphon sat up and rubbed its eyes: then it watched the Queen till she was out of sight: then it chuckled. “What fun!” said the Gryphon, half to itself, half to Alice."
    ]

    # Generate questions
    questions = generate_subjective_questions(input_texts)
    for i, question in enumerate(questions):
        print(f"Input {i + 1}: {input_texts[i]}")
        print(f"Generated Question {i + 1}: {question}\n")
