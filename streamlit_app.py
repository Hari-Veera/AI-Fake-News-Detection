import streamlit as st  # type: ignore
import pickle
import re
import nltk
import speech_recognition as sr # type: ignore
import pytesseract # type: ignore
from PIL import Image # type: ignore

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------- Setup --------------------
nltk.download('stopwords')
nltk.download('wordnet')

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -------------------- Functions --------------------
def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def predict_news(text):
    processed = preprocess(text)
    vector = vectorizer.transform([processed])
    return model.predict(vector)[0]

# -------------------- UI --------------------
st.set_page_config(page_title="AI Fake News Detection", layout="centered")

st.title("📰 AI Fake News Detection System")
st.caption("Give the news in any form. The system automatically processes it.")

input_type = st.radio(
    "How would you like to give the news?",
    ["Type or Paste Text", "Speak the News", "Upload News Image"],
    horizontal=True
)

# -------------------- TEXT INPUT (CHAT STYLE) --------------------
if input_type == "Type or Paste Text":
    user_input = st.chat_input("Type or paste news here...")

    if user_input:
        # Show user content
        with st.chat_message("user"):
            st.write(user_input)

        # Prediction
        result = predict_news(user_input)

        # Show system response
        with st.chat_message("assistant"):
            if result == "REAL":
                st.success("✅ This news is likely REAL")
            else:
                st.error("❌ This news is likely FAKE")

# -------------------- VOICE INPUT --------------------
elif input_type == "Speak the News":
    st.write("🎤 Click the button and speak clearly")

    if st.button("Start Speaking"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source)

        try:
            voice_text = recognizer.recognize_google(audio)

            # Show what user said
            with st.chat_message("user"):
                st.write(voice_text)

            result = predict_news(voice_text)

            with st.chat_message("assistant"):
                if result == "REAL":
                    st.success("✅ This news is likely REAL")
                else:
                    st.error("❌ This news is likely FAKE")

        except:
            st.error("Sorry, we could not understand the voice clearly.")

# -------------------- IMAGE INPUT --------------------
elif input_type == "Upload News Image":
    uploaded_image = st.file_uploader(
        "Upload an image that contains news text",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded image", use_column_width=True)

        extracted_text = pytesseract.image_to_string(image)

        if st.button("Check This News"):
            if len(extracted_text.strip()) == 0:
                st.warning("We could not find readable text in this image.")
            else:
                # Show extracted content
                with st.chat_message("user"):
                    st.write(extracted_text)

                result = predict_news(extracted_text)

                with st.chat_message("assistant"):
                    if result == "REAL":
                        st.success("✅ This news is likely REAL")
                    else:
                        st.error("❌ This news is likely FAKE")
