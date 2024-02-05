from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from joblib import load
from flask import Flask, jsonify, request
from flask_cors import CORS
from joblib import load
import torch
from torch import nn

class CustomDistilBERTModel(nn.Module):
    def __init__(self, distilbert, classification_head):
        super(CustomDistilBERTModel, self).__init__()
        self.distilbert = distilbert
        self.classification_head = classification_head

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state.mean(dim=1)  # Pooling over the sequence length
        predictions = self.classification_head(logits)
        return predictions




class ClassificationHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ClassificationHead, self).__init__()
        self.dropout1 = nn.Dropout(0.5)
        self.dense1 = nn.Linear(in_dim, 2048)
        self.dense2 = nn.Linear(2048,4096)
        self.dense3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.dense4 = nn.Linear(4096,2048)
        self.dense5 = nn.Linear(2048, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.dense6 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dropout2(x)
        x = self.dense6(x)
        return x


app = Flask(__name__)
CORS(app)

# Configure the database URI, change it based on your database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///feedbacks.db'
db = SQLAlchemy(app)

# Define a Feedback model for the database
class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    feedback = db.Column(db.String(255))
    pattern = db.Column(db.String(255))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models and tokenizers
presence_tokenizer = load('/Dark-Patterns/dark-patterns-recognition/api/presence_tokenizer_distilbert.joblib')
presence_model = load('/Dark-Patterns/dark-patterns-recognition/api/presence_classifier_distilbert.joblib')
category_model = load('/Dark-Patterns/dark-patterns-recognition/api/category_classifier_bert.joblib')
category_tokenizer = load('/Dark-Patterns/dark-patterns-recognition/api/category_tokenizer_bert.joblib')

classes = ["Social Proof", "Misdirection", "Urgency", "Forced Action", "Obstruction", "Sneaking", "Scarcity"]
blacklist = ['customer care', 'view more', 'stuffed animals','₹','$15.99','prime','subtotal','Brand','Model Name','Operating System','1 TB','secure transaction','en','add gift options','emi','all', ]
social=['ratings']

@app.route('/', methods=['POST'])
def main():
    if request.method == 'POST':
        output = []
        data = request.get_json().get('tokens')

        for token in data:
            token_lower = token.lower()
            # if any(word in token_lower for word in blacklist):
            #     continue
            input_ids = presence_tokenizer.encode(token, return_tensors='pt')
            attention_mask = torch.ones_like(input_ids)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = presence_model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                confidence_score = torch.nn.functional.softmax(outputs, dim=1)[:, 1].item()

            # Set a confidence threshold for presence classification
            presence_confidence_threshold = 0

            if predictions == 1 and confidence_score >= presence_confidence_threshold:
                result = 'Dark'
            else:
                result = 'Not Dark' # Not Dark

            if result == 'Dark':
                input_encoding = category_tokenizer(token, truncation=True, padding=True, return_tensors='pt')
                input_encoding = input_encoding.to(device)

                with torch.no_grad():
                    logits = category_model(**input_encoding).squeeze()

                    # Apply softmax to get probabilities
                    probabilities = torch.nn.functional.softmax(logits, dim=0)

                    # Get the predicted class and its corresponding confidence score
                    predicted_class = torch.argmax(probabilities).item()
                    confidence_score = probabilities[predicted_class].item()

                    # Set a confidence threshold for category classification
                    category_confidence_threshold = 0
                    class_value = classes[predicted_class]
                    # Check if confidence exceeds the threshold
                    if confidence_score >= category_confidence_threshold:
                        output.append(class_value)

            else:
                output.append(result)

        dark = [data[i] for i in range(len(output)) if output[i] == 'Dark']
        # for d in dark:
        #     print(d)
        # print()
        # print(len(dark))

        message = '{ \'result\': ' + str(output) + ' }'
        # print(message)

        json = jsonify(message)

        return json
    
@app.route('/', methods=['GET'])
def handle_get_request():
    message = {'result': 'hello world'}
    return jsonify(message)





def handle_feedback_post_request():
    feedback_data = request.get_json()
    feedback = feedback_data.get('feedback')
    pattern = feedback_data.get('pattern')

    # Save feedback to the local database
    new_feedback = Feedback(feedback=feedback, pattern=pattern)
    db.session.add(new_feedback)
    db.session.commit()

    print(f"Received Feedback: {feedback}, Pattern: {pattern}")

    return jsonify({"message": "Feedback received successfully"})


@app.route('/feedback', methods=['GET'])
def get_all_feedback():
    # Create an application context
    with app.app_context():
        # Query all feedback entries from the database
        feedback_entries = Feedback.query.all()
        print(feedback_entries)

        # Convert feedback entries to a list of dictionaries
        feedback_list = [{'id': entry.id, 'feedback': entry.feedback, 'pattern': entry.pattern} for entry in feedback_entries]

    return jsonify({"feedback": feedback_list})

if __name__ == '__main__':
    # Create the database tables before running the app
    with app.app_context():
        db.create_all()
    app.run(threaded=True, debug=True)