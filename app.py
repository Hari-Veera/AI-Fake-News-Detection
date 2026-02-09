import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

print("===================================")
print(" AI Fake News Detection System")
print(" Type 'exit' to quit")
print("===================================")

while True:
    try:
        news = input("\nEnter News Text: ")
    except EOFError:
        print("\nInput error detected. Please restart the program.")
        break

    if news.lower() == "exit":
        print("Exiting application...")
        break

    if len(news.strip()) == 0:
        print("Please enter valid news text.")
        continue

    processed_text = preprocess(news)
    vector = vectorizer.transform([processed_text])
    prediction = model.predict(vector)[0]

    print("Prediction:", prediction)

