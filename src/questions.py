from transformers import pipeline
import pandas as pd

# Load your CSV
input_file_path = r'C:/Users/ozdep/Documents/ltp final project/LTP/data/processed/cleaned_text.csv'
output_file_path = r'C:/Users/ozdep/Documents/ltp final project/LTP/data/processed/alicesegments.csv'

# Read the file as a single string
with open(input_file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Split the content into paragraphs using double newlines as separators
paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

# Use the google/flan-t5-xl model for question generation
question_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-xl",
    device_map="auto"
)

# Initialize a list to store the data
data = []

# Define the question words
question_words = ["What", "Why", "How"]

# Function to generate a question starting with a specific word
def generate_question_with_word(paragraph, question_word, max_retries=3):
    for _ in range(max_retries):
        questions = question_generator(
            f"Generate a question that requires a personal opinion and starts with '{question_word}': {paragraph}",
            max_length=300,
            num_return_sequences=1,
            clean_up_tokenization_spaces=True
        )
        generated_question = questions[0]['generated_text']
        # Ensure the generated question starts with the desired word
        if generated_question.strip().lower().startswith(question_word.lower()):
            return generated_question
    # If retries are exhausted, return the last generated question
    return generated_question

# Generate questions for each paragraph, alternating question words
for index, paragraph in enumerate(paragraphs):
    question_word = question_words[index % len(question_words)]
    generated_question = generate_question_with_word(paragraph, question_word)
    print(f"Question for paragraph {index + 1}: {generated_question}")

    # Append to the data list as a dictionary
    data.append({
        "source-text": paragraph,
        "target-text": generated_question
    })

# Convert the data list to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(output_file_path, index=False, encoding='utf-8')
print(f"Generated questions saved to {output_file_path}")
