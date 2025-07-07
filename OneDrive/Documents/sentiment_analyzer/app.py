from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline
from pymongo import MongoClient
from datetime import datetime

# Initialize app
app = Flask(__name__)

# Sentiment pipeline using Transformers
sentiment_model = pipeline("sentiment-analysis")

# MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")
db = client["sentimentDB"]
collection = db["results"]

# HTML Template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Sentiment Analyzer</title>
</head>
<body>
    <h1>AI-Based Sentiment Analyzer</h1>
    <form method="POST" action="/analyze">
        <textarea name="text" rows="5" cols="60" placeholder="Enter review or text here..."></textarea><br><br>
        <input type="submit" value="Analyze">
    </form>
    {% if sentiment %}
        <h2>Sentiment: {{ sentiment['label'] }} ({{ sentiment['score']*100 | round(2) }}%)</h2>
    {% endif %}
    <hr>
    <h3>Previous Results</h3>
    <ul>
    {% for entry in history %}
        <li><b>{{ entry['text'] }}</b> → {{ entry['result']['label'] }} ({{ (entry['result']['score']*100)|round(2) }}%)</li>
    {% endfor %}
    </ul>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    history = list(collection.find().sort("timestamp", -1).limit(10))
    return render_template_string(html_template, sentiment=None, history=history)

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    if not text.strip():
        return jsonify({"error": "Empty text"}), 400
    
    result = sentiment_model(text)[0]
    record = {
        "text": text,
        "result": result,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(record)
    history = list(collection.find().sort("timestamp", -1).limit(10))
    return render_template_string(html_template, sentiment=result, history=history)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Text is required"}), 400
    result = sentiment_model(text)[0]
    collection.insert_one({
        "text": text,
        "result": result,
        "timestamp": datetime.utcnow()
    })
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
