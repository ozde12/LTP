import re

# File paths
input_file_path = r"C:\Users\ozdep\Documents\ltp final project\LTP\data\raw\trial.txt"
output_file_path = r"C:\Users\ozdep\Documents\ltp final project\LTP\data\processed\cleaned_text.csv"

# Function to clean the file and process text
def process_text_file(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        cleaned_text = []
        current_paragraph = []
        skip_next_line = False
        is_dialogue = False

        for line in lines:
            # Remove unnecessary characters like * and _
            cleaned_line = re.sub(r'[\*_]', '', line).strip()

            # Check for chapter headings (e.g., "CHAPTER X.")
            if re.match(r'^CHAPTER \w+\.', cleaned_line, re.IGNORECASE):
                skip_next_line = True  # Skip the next line (title of the chapter)
                continue

            # Skip the line if it follows a chapter heading
            if skip_next_line:
                skip_next_line = False
                continue

            # Detect dialogues (lines that start and end with quotes or include common dialogue keywords)
            if re.match(r'^".*"$', cleaned_line) or re.search(r'\b(said|replied|asked|exclaimed|shouted)\b', cleaned_line, re.IGNORECASE):
                is_dialogue = True

            # Add non-empty lines to the current paragraph
            if cleaned_line:
                current_paragraph.append(cleaned_line)
            else:
                # Save the completed paragraph or dialogue block
                if current_paragraph:
                    if is_dialogue:
                        # Group dialogue lines into one block
                        cleaned_text.append([' '.join(current_paragraph)])
                        is_dialogue = False
                    else:
                        cleaned_text.append(current_paragraph)
                    current_paragraph = []

        # Add the last paragraph if it exists
        if current_paragraph:
            cleaned_text.append(current_paragraph)

        # Write the cleaned text to the output file
        with open(output_path, 'w', encoding='utf-8') as file:
            for paragraph in cleaned_text:
                for line in paragraph:
                    file.write(line + "\n")
                file.write("\n")  # Add a blank line between paragraphs

        print(f"File processed and saved successfully: {output_path}")

    except FileNotFoundError:
        print(f"Input file not found: {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Clean and process the file
process_text_file(input_file_path, output_file_path)
