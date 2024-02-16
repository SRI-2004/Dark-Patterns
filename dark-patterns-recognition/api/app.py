from flask import Flask, jsonify, request
from flask_cors import CORS
from joblib import load
from pydantic import BaseModel
from flask_sqlalchemy import SQLAlchemy

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

presence_classifier = load('C:/Users/crazax/Downloads/dark-patterns-recognition/dark-patterns-recognition/api/presence_classifier.joblib')
presence_vect = load('C:/Users/crazax/Downloads/dark-patterns-recognition/dark-patterns-recognition/api/presence_vectorizer.joblib')
category_classifier = load('C:/Users/crazax/Downloads/dark-patterns-recognition/dark-patterns-recognition/api/category_classifier.joblib')
category_vect = load('C:/Users/crazax/Downloads/dark-patterns-recognition/dark-patterns-recognition/api/category_vectorizer.joblib')



@app.route('/', methods=['POST'])
def main():
    if request.method == 'POST':
        output = []
        data = request.get_json().get('tokens')
        print(len(data))

        for token in data:
            result = presence_classifier.predict(presence_vect.transform([token]))
            if result == 'Dark':
                cat = category_classifier.predict(category_vect.transform([token]))
                output.append(cat[0])
                print(token,cat[0])
            else:
                output.append(result[0])

        dark = [data[i] for i in range(len(output)) if output[i] == 'Dark']
        for d in dark:
            print(d)
        print()
        print(len(dark))

        message = '{ \'result\': ' + str(output) + ' }'
        print(len(output))
       # print(message)

        json = jsonify(message)

        return json
    
@app.route('/', methods=['GET'])
def handle_get_request():
    message = {'result': 'hello world'}
    return jsonify(message)



@app.route('/feedback', methods=['POST'])
def handle_feedback_post_request():
    feedback_data = request.get_json()
    feedback = feedback_data.get('feedback')
    pattern = feedback_data.get('pattern')

    # Create an application context
    with app.app_context():
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

        # Convert feedback entries to a list of dictionaries
        feedback_list = [{'id': entry.id, 'feedback': entry.feedback, 'pattern': entry.pattern} for entry in feedback_entries]

    return jsonify({"feedback": feedback_list})

if __name__ == '__main__':
    # Create the database tables before running the app
    with app.app_context():
        db.create_all()
    app.run(threaded=True, debug=True)