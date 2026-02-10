import streamlit as st
import pickle
import re
import nltk
import os
import pytesseract
from PIL import Image

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------------------- Safe Voice Import --------------------
try:
    import speech_recognition as sr
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="AI Fake News Detection",
    layout="centered"
)

# -------------------- NLTK --------------------
nltk.download("stopwords")
nltk.download("wordnet")

# -------------------- Model Load --------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -------------------- Session State --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- Functions --------------------
def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def predict_news(text):
    processed = preprocess(text)
    vector = vectorizer.transform([processed])
    return model.predict(vector)[0]

# -------------------- UI --------------------
st.title("📰 AI Fake News Detection System")
st.caption("Give the news in any form. The system does everything automatically.")

options = ["Type or Paste Text", "Upload News Image"]
if VOICE_ENABLED:
    options.insert(1, "Speak the News")

input_type = st.radio(
    "How would you like to give the news?",
    options,
    horizontal=True
)

# -------------------- Chat History --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------- TEXT INPUT --------------------
if input_type == "Type or Paste Text":
    user_input = st.chat_input("Type or paste news here...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        result = predict_news(user_input)
        reply = "✅ This news is likely REAL" if result == "REAL" else "❌ This news is likely FAKE"

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )
        st.rerun()

# -------------------- VOICE INPUT --------------------
elif input_type == "Speak the News":
    st.warning("🎤 Voice input works only in local environment, not on cloud.")

# -------------------- IMAGE INPUT --------------------
elif input_type == "Upload News Image":
    uploaded_image = st.file_uploader(
        "Upload an image containing news text",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        extracted_text = pytesseract.image_to_string(image)

        if st.button("Analyze Image"):
            if extracted_text.strip() == "":
                st.warning("⚠️ No readable text found in image.")
            else:
                st.session_state.messages.append(
                    {"role": "user", "content": f"🖼️ {extracted_text}"}
                )

                result = predict_news(extracted_text)
                reply = "✅ This news is likely REAL" if result == "REAL" else "❌ This news is likely FAKE"

                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )
                st.rerun()



