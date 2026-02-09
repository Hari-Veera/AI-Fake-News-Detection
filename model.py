import pandas as pd # type: ignore
import nltk # type: ignore
import re
import pickle

from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore

from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Add labels
true_df["label"] = "REAL"
fake_df["label"] = "FAKE"

# Combine datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Keep only required columns
df = df[["text", "label"]]

# Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df["text"] = df["text"].apply(preprocess)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
