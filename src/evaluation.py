import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from evaluate import load

def main():
    model_path = "./trained_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df = pd.read_csv("data/processed/alicesegments.csv")

    dataset = Dataset.from_pandas(df)
    dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = dataset_split["test"]

    sources = test_dataset["source-text"]
    references = test_dataset["target-text"]

    # generate predictions
    predictions = []
    for src in sources:
        # tokenize the input
        inputs = tokenizer(src, return_tensors="pt", truncation=True, max_length=512)
        for k,v in inputs.items():
            inputs[k] = v.to(device)

        # generate
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=50)
        # decode
        pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred_text)


    bleu_metric = load("bleu")
    rouge_metric = load("rouge")
    bertscore_metric = load("bertscore")



    references_for_bleu = [[r] for r in references]

    bleu_results = bleu_metric.compute(
        predictions=predictions,
        references=references_for_bleu
    )

    rouge_results = rouge_metric.compute(
        predictions=predictions,
        references=references
    )

    bertscore_results = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        model_type="bert-base-uncased"
    )


    print("===== EVALUATION RESULTS =====\n")

    # BLEU
    print(f"BLEU: {bleu_results['bleu']:.4f}")

    # ROUGE
    print("ROUGE-1 F1:", f"{rouge_results['rouge1']:.4f}")
    print("ROUGE-2 F1:", f"{rouge_results['rouge2']:.4f}")
    print("ROUGE-L F1:", f"{rouge_results['rougeL']:.4f}")

    # BERTScore
    bert_p = sum(bertscore_results["precision"])/len(bertscore_results["precision"])
    bert_r = sum(bertscore_results["recall"])/len(bertscore_results["recall"])
    bert_f = sum(bertscore_results["f1"])/len(bertscore_results["f1"])
    print(f"BERTScore (avg): P={bert_p:.4f} R={bert_r:.4f} F1={bert_f:.4f}")

    print("\n================================")

if __name__ == "__main__":
    main()
