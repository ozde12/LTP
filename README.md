# Subjective Question Generation for Narrative Texts

This project focuses on generating subjective questions from narrative texts, using Lewis Carroll's *Alice in Wonderland* as the primary dataset. The project uses the Flan-T5-small model to generate questions that elicit opinions, interpretations, and personal reflections.

### Goals
- Generate high-quality subjective questions from literary texts.
- Make model be able to handle narrative text challenges such as complex plots, varied styles, and character development.
- Evaluate the model using traditional metrics BLEU, ROUGE and semantic metrics BERTScore.

### Features
- Fine-tuned Flan-T5 model.
- Comprehensive evaluation with both automated metrics and qualitative assessments.
- Data preprocessing tailored for narrative texts.


Install dependencies with:
```bash

pip install -r requirements.txt
```

## Usage

### Data Preprocessing

Run the following script to preprocess the raw text data:

```bash
python LTP/src/clean_data.py
```
This script cleans the text from *Alice in Wonderland* and prepares it in CSV format for training.

### Model Training

Fine-tune the model using:

```bash
python LTP/src/model_training.py
```
The script will:
- Load and preprocess the dataset.
- Fine-tune the Flan-T5-large model.
- Save the model and tokenizer to `LTP/outputs/final_model`.

### Evaluation

Evaluate the model's performance with:

```bash
python LTP/src/evaluation.py
```
This generates the evaluation metrics (e.g., BLEU, ROUGE, BERTScore) and saves a bar plot of the metrics as `evaluation_metrics.png`.

---

## Results

### Evaluation Metrics
| Metric         | Score  |
|----------------|--------|
| BLEU           | 0.1488 |
| ROUGE-1 F1     | 0.3681 |
| ROUGE-2 F1     | 0.2245 |
| ROUGE-L F1     | 0.3613 |
| BERTScore F1   | 0.6179 |


## Key Files

- **`clean_data.py`**: Prepares the text data for training.
- **`model_training.py`**: Fine-tunes the Flan-T5-small model.
- **`evaluation.py`**: Evaluates the model and generates metrics.
