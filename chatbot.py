from collections import defaultdict
from nltk.tokenize import word_tokenize

# Symptom dictionary to handle variations in user input
symptom_dict = {
    'fever': 'Fever',
    'headache': 'Headache',
    'cough': 'Cough',
    'fatigue': 'Fatigue',
    'chest pain': 'Chest Pain',
    'shortness of breath': 'Short Breath',
    'short breath': 'Short Breath',
    'weight loss': 'Weight Loss',
    'sweating': 'Sweating',
    'nausea': 'Nausea',
    'vomiting': 'Vomiting',
    'dizziness': 'Dizziness',
    # ... add more symptoms as needed
}

# Simple chatbot
def basic_chatbot():
    # Dictionary of symptoms and diseases
    symptoms_data = {
        'Symptom 1': ['Fever', 'Fatigue', 'Chest Pain', 'Fever', 'Nausea', 'Headache', 'Fever', 'Cough', 'Chest Pain', 'Fatigue'],
        'Symptom 2': ['Headache', 'Cough', 'Short Breath', 'Cough', 'Vomiting', 'Dizziness', 'Sweating', 'Short Breath', 'Dizziness'],
        'Symptom 3': ['Cough', 'Weight Loss', 'Sweating', 'Sore Throat', 'Fatigue', 'Blurred Vision', 'Weight Loss', 'Fatigue', 'Headache'],
        'Disease': ['Flu', 'Tuberculosis', 'Heart Disease', 'Common Cold', 'Gastroenteritis', 'Migraine', 'Diabetes', 'Asthma', 'Hypertension'],
    }

    print('Hello! I am a health chatbot. Tell me your symptoms, and I\'ll try to predict a possible disease.')
    while True:
        user_input = input("You: ").strip().lower()

        if "hello" in user_input or "hi" in user_input:
            print("Bot: Hello! How can I assist you with your health concerns today?")
        elif "how are you" in user_input:
            print("Bot: I'm just a bot, but I'm here to help you!")
        elif "bye" in user_input or "goodbye" in user_input or "exit" in user_input or "by" in user_input:
            print("Bot: Goodbye! Take care and stay healthy!")
            break
        elif "name" in user_input:
            print("Bot: I am your health assistant chatbot.")
        else:
            # symptom processing
            tokens = word_tokenize(user_input)
            symptoms_found = []
            for token in tokens:
                if token in symptom_dict:
                    symptoms_found.append(symptom_dict[token])

            # Check for potential diseases
            disease_counts = defaultdict(int)
            total_matches = 0
            for i in range(len(symptoms_data['Disease'])):
                # Count matching symptoms for each disease
                matching_symptoms = sum(1 for symptom in [symptoms_data['Symptom 1'][i].lower(),
                                                          symptoms_data['Symptom 2'][i].lower(),
                                                          symptoms_data['Symptom 3'][i].lower()] if symptom in user_input)

                if matching_symptoms > 0:
                    disease_counts[symptoms_data['Disease'][i]] = matching_symptoms
                    total_matches += matching_symptoms

            if total_matches > 0:
                # Calculate and print percentage chances for each disease
                print("Bot: Based on your symptoms, here are the possible diseases and their likelihood:")
                for disease, count in disease_counts.items():
                    percentage = (count / total_matches) * 100
                    print(f"- {disease}: {percentage:.2f}%")
                print("Please consult a doctor for a proper diagnosis.")
            else:
                print("Bot: I'm sorry, I couldn't identify a likely disease based on the symptoms provided. Can you please provide more information?")

if __name__ == "__main__":
    basic_chatbot()
