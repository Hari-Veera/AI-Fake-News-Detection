import streamlit as st
import pickle
import re
import nltk
import speech_recognition as sr
import pytesseract
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------- NLTK Setup --------------------
nltk.download("stopwords")
nltk.download("wordnet")

# -------------------- Load Model --------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -------------------- Helper Functions --------------------
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

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="AI Fake News Detection",
    page_icon="🔍",
    layout="centered"
)

# -------------------- UI --------------------
st.markdown("## 🔍 AI Fake News Detection")
st.caption(
    "DL classifier for detecting fake news using text, voice, or images."
)

input_type = st.radio(
    "Select input method:",
    ["Text", "Voice", "Image"],
    horizontal=True
)

# -------------------- TEXT INPUT --------------------
if input_type == "Text":
    user_input = st.chat_input("Enter a news headline or article...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        result = predict_news(user_input)

        with st.chat_message("assistant"):
            if result == "REAL":
                st.success("✅ This news is likely REAL")
            else:
                st.error("❌ This news is likely FAKE")

# -------------------- VOICE INPUT --------------------
elif input_type == "Voice":
    st.info("Click the button and speak clearly (English).")

    if st.button("🎤 Start Recording"):
        recognizer = sr.Recognizer()

        try:
            with sr.Microphone() as source:
                st.info("Listening...")
                audio = recognizer.listen(source, timeout=5)

            voice_text = recognizer.recognize_google(audio)

            with st.chat_message("user"):
                st.write(voice_text)

            result = predict_news(voice_text)

            with st.chat_message("assistant"):
                if result == "REAL":
                    st.success("✅ This news is likely REAL")
                else:
                    st.error("❌ This news is likely FAKE")

        except sr.WaitTimeoutError:
            st.error("⏱️ Listening timed out. Try again.")
        except sr.UnknownValueError:
            st.error("❌ Could not understand the voice.")
        except Exception:
            st.error("⚠️ Voice input not supported in this environment.")

# -------------------- IMAGE INPUT --------------------
elif input_type == "Image":
    uploaded_image = st.file_uploader(
        "Upload an image containing news text",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        extracted_text = pytesseract.image_to_string(image)

        if st.button("Check This News"):
            if extracted_text.strip() == "":
                st.warning("⚠️ No readable text found in the image.")
            else:
                with st.chat_message("user"):
                    st.write(extracted_text)

                result = predict_news(extracted_text)

                with st.chat_message("assistant"):
                    if result == "REAL":
                        st.success("✅ This news is likely REAL")
                    else:
                        st.error("❌ This news is likely FAKE")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("🚀 Project by Hari Veera | AI Fake News Detection System")

