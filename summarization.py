from transformers import T5Tokenizer, T5ForConditionalGeneration
import shap

# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize_text(text, max_length=500):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    
    # Tokenize input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=700, truncation=True)
    
    # Generate summary
    outputs = model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # SHAP Explainer (optional if explainability is required)
    explainer = shap.Explainer(model, masker=shap.maskers.Text(tokenizer))
    
    # Interpret outputs (ensure proper format for SHAP)
    shap_values = explainer([text])  # Use raw text input for SHAP explainer
    
    return summary, shap_values
