from flask import Flask, request, jsonify
import fitz  # PyMuPDF for PDF processing
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from config import OPENAI_API_KEY

app = Flask(__name__)

# Initialize OpenAI model
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4")

# PDF text extraction function
def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page in pdf_document:
        text += page.get_text()
    return text

# Split text into chunks for better LLM processing
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Route for PDF upload and summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400
    
    pdf_file = request.files['pdf']
    pdf_path = os.path.join("/tmp", pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract and chunk PDF text
    pdf_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pdf_text)

    # Summarize each chunk
    summaries = [llm.invoke(chunk).content for chunk in chunks]

    final_summary = " ".join(summaries)
    return jsonify({"summary": final_summary})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
