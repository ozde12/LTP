# Subjective Question Generation for Narrative Texts

This project focuses on generating subjective questions from narrative texts, using Lewis Carroll's *Alice in Wonderland* as the primary dataset. The project extends the capabilities of "Opinerium," which fine-tunes the Flan-T5-large model to generate questions that elicit opinions, interpretations, and personal reflections.

### Goals
- Generate high-quality subjective questions from literary texts.
- Make model be able to handle narrative text challenges such as complex plots, varied styles, and character development.
- Evaluate the model using both traditional metrics (e.g., BLEU, ROUGE) and semantic metrics (e.g., BERTScore).

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
python LTP/src/data_preprocessing.py
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

### Visual Representation
See `evaluation_metrics.png` for a bar plot visualizing the metric scores.

---

## Key Files

- **`data_preprocessing.py`**: Prepares the text data for training.
- **`model_training.py`**: Fine-tunes the Flan-T5-large model.
- **`evaluation.py`**: Evaluates the model and generates metrics.

---

## Future Work
- Expand the dataset to include other narrative texts.
- Implement multilingual subjective question generation.
- Explore alternative transformer models for comparative analysis.

---

## Citation
If you use this repository, please cite:

```plaintext
Babakhani, P., Lommatzsch, A., Brodt, T., et al. (2024). "Opinerium: Subjective Question Generation using Large Language Models." IEEE Access.
