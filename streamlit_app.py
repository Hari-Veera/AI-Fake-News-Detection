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
st.set_page_config(
    page_title="AI Fake News Detection System",
    layout="centered"
)

st.title("📰 AI Fake News Detection System")
st.caption("Give the news in any form. The system does everything automatically.")

input_type = st.radio(
    "How would you like to give the news?",
    ["Type or Paste Text", "Speak the News", "Upload News Image"],
    horizontal=True
)

# -------------------- TEXT INPUT (BOTTOM CHAT INPUT) --------------------
if input_type == "Type or Paste Text":
    st.markdown("✍️ **Type or paste the news below**")

    user_input = st.chat_input("Type or paste news here...")

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
elif input_type == "Speak the News":
    st.markdown("🎤 **Click the button and speak clearly**")

    if st.button("Start Speaking"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source)

        try:
            voice_text = recognizer.recognize_google(audio)

            with st.chat_message("user"):
                st.write(voice_text)

            result = predict_news(voice_text)

            with st.chat_message("assistant"):
                if result == "REAL":
                    st.success("✅ This news is likely REAL")
                else:
                    st.error("❌ This news is likely FAKE")

        except:
            st.error("Sorry, could not understand the voice clearly.")

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
            if extracted_text.strip() == "":
                st.warning("No readable text found in the image.")
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

