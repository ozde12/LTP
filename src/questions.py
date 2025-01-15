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
    model="google/flan-t5-xl"
)

# Initialize a list to store the data
data = []

# Generate questions for each paragraph
for index, paragraph in enumerate(paragraphs):
    questions = question_generator(
        f"Write an open-enden question that requires a personal opinion: {paragraph}",
        max_length=300,
        num_return_sequences=1,
        clean_up_tokenization_spaces=True
    )
    generated_question = questions[0]['generated_text']
    print(f"Subjective question for paragraph {index + 1}: {generated_question}")

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
