from operator import le
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from flask import Flask, request, jsonify, render_template # Import render_template
import random

df = pd.read_csv('dataset.csv')
X = df[['Symptom_1', 'Symptom_2', 'Symptom_3']]
y = df['Disease']

# 1. Create a LabelEncoder object for features
le_X = LabelEncoder()

# 2. Encode categorical features
for column in X.columns:
    X[column] = le_X.fit_transform(X[column])

# 3. Encode the target variable
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Create and train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Sample symptom dictionary to convert user input into recognizable symptoms
symptom_dict = {
    'fever': 'Fever',
    'headache': 'Headache',
    'cough': 'Cough',
    'fatigue': 'Fatigue',
    'chest pain': 'Chest Pain',
    'shortness of breath': 'Short Breath'
}

def process_input(user_input):
    tokens = word_tokenize(user_input.lower())
    symptoms_list = [symptom_dict.get(token) for token in tokens if token in symptom_dict]
    while len(symptoms_list) < 3:
        symptoms_list.append(None)
    return symptoms_list[:3]

app = Flask(__name__)

# Define Possible responses
responses = {
    'greeting': ["Hi! How can I assist you today?", "Hello! What symptoms are you experiencing?", "Hey there! How can I help you today?"],
    'goodbye': ["Goodbye! Stay healthy!", "Take care!"],
    'symptoms': ["Please tell me your symptoms.", "What symptoms are you feeling?"],
    'unknown': ["I'm not sure about that. Can you tell me your symptoms again?"]
}

# Disease prediction function using the trained model
def predict_disease(symptoms):
    encoded_symptoms = []
    for symptom in symptoms:
        if symptom in le_X.classes_:
            encoded_symptoms.append(le_X.transform([symptom])[0])
        else:
            print(f"Warning: Symptom '{symptom}' not found in training data.")
            encoded_symptoms.append(-1)

    if len(encoded_symptoms) == 3:
        input_data = pd.DataFrame([encoded_symptoms], columns=['Symptom_1', 'Symptom_2', 'Symptom_3'])
        prediction = model.predict(input_data)
        predicted_disease_encoded = prediction[0]
        predicted_disease = le_y.inverse_transform([predicted_disease_encoded])[0]
        return predicted_disease
    else:
        return "Could not process symptoms for prediction."

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Chatbot Route
@app.route('/chat', methods=['POST']) # Changed to POST as the frontend sends a POST request
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"response": "Invalid input"}), 400

    user_input = data['message']

    if "hello" in user_input.lower() or "hi" in user_input.lower():
        return jsonify({"response": random.choice(responses['greeting'])})
    elif "bye" in user_input.lower():
        return jsonify({"response": random.choice(responses['goodbye'])})
    else:
        symptoms = process_input(user_input)
        if any(s is not None for s in symptoms):
            disease = predict_disease(symptoms)
            response = f"Based on your symptoms {', '.join(filter(None, symptoms))}, you might have {disease}. Please consult a doctor for a proper diagnosis."
        else:
            response = random.choice(responses['unknown'])

        return jsonify({"response": response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)