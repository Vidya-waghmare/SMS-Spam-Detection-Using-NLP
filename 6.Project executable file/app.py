from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load and preprocess dataset
data = pd.read_csv('spam_ham_dataset.csv')  # Replace with the path to your dataset
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label_num']

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']  # Get input from form
    input_vectorized = vectorizer.transform([input_text])  # Vectorize input
    prediction = model.predict(input_vectorized)[0]  # Make prediction
    label = "Spam" if prediction == 1 else "Ham"  # Convert to label

    return render_template('spam.html', prediction=label, text=input_text)

if __name__ == '__main__':
    app.run(debug=True)

