from flask import Flask, request, render_template, jsonify
import os
from scripts.pdf_extraction import extract_text_from_pdf
from scripts.summarization import summarize_text
from scripts.explainability import generate_explanation

app = Flask(__name__)

# Ensure required directories exist
os.makedirs("data/uploaded_pdfs", exist_ok=True)
os.makedirs("data/summaries", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    pdf_file = request.files["pdf"]
    if not pdf_file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the uploaded file
    pdf_path = os.path.join("data/uploaded_pdfs", pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract text and generate summary
    pdf_text = extract_text_from_pdf(pdf_path)
    summary = summarize_text(pdf_text)

    # Generate explainability data (word importance)
    explanation = generate_explanation(summary)

    # Save the summary and explanation
    summary_path = os.path.join("data/summaries", f"{pdf_file.filename}_summary.txt")
    with open(summary_path, "w") as file:
        file.write(summary)

    return jsonify({"summary": summary, "explanation": explanation})


if __name__ == "__main__":
    app.run(debug=True)
