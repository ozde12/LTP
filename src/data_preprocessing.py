import os
import pandas as pd
from nltk.tokenize import sent_tokenize

def preprocess_text(input_path, output_path):
    """
    Preprocesses text files by segmenting into smaller units (e.g., paragraphs).
    Args:
    
        input_path (str): Path to the raw text file.
        output_path (str): Path to save the processed CSV file.
    """
    with open(input_path, 'r', encoding="utf-8") as file:
        text = file.read()

    # Tokenize text into sentences
    segments = sent_tokenize(text)

    # Save as a DataFrame
    df = pd.DataFrame({'segments': segments})
    df.to_csv(output_path, index=False)
    print(f"Processed text saved to {output_path}")

if __name__ == "__main__":
    # File paths for Alice in Wonderland
    input_file = "data/raw/alice.txt"
    output_file = "data/processed/alice_segments.csv"

    # Preprocess text
    preprocess_text(input_file, output_file)
