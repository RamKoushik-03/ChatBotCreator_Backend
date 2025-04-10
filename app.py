from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os


app = Flask(__name__)
CORS(app)  

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


BUSINESS_INFO_FILE = "business_info.txt"

def read_business_info():
    """
    Reads business info from the text file.
    """
    if not os.path.exists(BUSINESS_INFO_FILE):
        return ""
    with open(BUSINESS_INFO_FILE, "r", encoding="utf-8") as file:
        return file.read().strip()

def write_business_info(info):
    """
    Writes business info to the text file.
    """
    if not info:
        raise ValueError("Business information cannot be empty.")
    with open(BUSINESS_INFO_FILE, "w", encoding="utf-8") as file:
        file.write(info)

@app.route('/submit-info', methods=['POST'])
def submit_info():
    """
    Endpoint to submit business information.
    """
    try:
        data = request.get_json()
        business_info = data.get('info', 'temp-1').strip()
        if not business_info:
            return jsonify({'error': 'Business information cannot be empty.'}), 400

        
        write_business_info(business_info)
        print(f"Business info saved: {business_info}")
        return jsonify({'message': 'Business information saved successfully!'}), 200
    except ValueError as ve:
        print(f"Validation Error in /submit-info: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error in /submit-info: {e}")
        return jsonify({'error': 'An error occurred while saving business information.'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint to handle user questions based on submitted business info.
    """
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'Question is required.'}), 400

        
        business_info = read_business_info()
        if not business_info:
            return jsonify({'error': 'Business information not found. Please submit it first.'}), 400

        print(f"Question received: {question}")
        print(f"Using business info: {business_info}")

        
        result = qa_pipeline(question=question, context=business_info)
        confidence_threshold = 0.1
        if result['score'] < confidence_threshold:
            answer = "I'm sorry, I couldn't find a specific answer. Please contact support for further assistance."
        else:
            answer = result['answer']

        return jsonify({'answer': answer}), 200
    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({'error': 'An error occurred while processing your question.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
