from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "./trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Example: let's test generating a question from a new snippet
input_text = """There might be some sense in your knocking,” the Footman went on without attending to her, “if we had the door between us. For instance, if you were nside,you might knock, and I could let you out, you know.” He was looking up into the sky all the time he was speaking, and this Alice thought decidedly uncivil. “But perhaps he can’t help it,” she said to herself; “his eyes are so ery early at the top of his head. But at any rate he might answer questions.—How am I to get in?” she repeated, aloud."""
inputs = tokenizer(input_text, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=50)
question = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated question:", question)
