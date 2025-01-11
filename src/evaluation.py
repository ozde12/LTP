from datasets import load_metric
from transformers import pipeline

def evaluate_generated_questions(references, predictions):
    """
    Evaluates the quality of generated questions using BLEU, ROUGE, and BERTScore.
    
    Args:
        references (list): List of reference questions.
        predictions (list): List of generated questions.

    Returns:
        dict: Dictionary containing evaluation scores.
    """
    # Load metrics
    bleu_metric = load_metric("bleu")
    rouge_metric = load_metric("rouge")
    bertscore = pipeline("text-classification", model="microsoft/deberta-xlarge-mnli")

    # BLEU
    bleu_score = bleu_metric.compute(predictions=[p.split() for p in predictions],
                                      references=[[r.split()] for r in references])

    # ROUGE
    rouge_scores = rouge_metric.compute(predictions=predictions, references=references)

    # BERTScore
    bert_scores = bertscore(predictions, references)

    # Results
    evaluation_results = {
        "BLEU": bleu_score["bleu"],
        "ROUGE-1": rouge_scores["rouge1"].mid.fmeasure,
        "ROUGE-2": rouge_scores["rouge2"].mid.fmeasure,
        "ROUGE-L": rouge_scores["rougeL"].mid.fmeasure,
        "BERTScore": sum([score["score"] for score in bert_scores]) / len(bert_scores)
    }
    return evaluation_results

if __name__ == "__main__":
    # Example references and predictions
    references = [
        "What do you think Mr. Dursley felt when he saw the cat reading a map?",
        "How does Harry feel about being a wizard?"
    ]
    predictions = [
        "How do you think Mr. Dursley felt when he saw the cat reading a map?",
        "What are Harry's thoughts on becoming a wizard?"
    ]

    # Evaluate
    scores = evaluate_generated_questions(references, predictions)
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
