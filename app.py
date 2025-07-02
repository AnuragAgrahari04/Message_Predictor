import streamlit as st
import numpy as np
import pickle
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Page Config ---
st.set_page_config(page_title="✍️ LSTM Text Generator", page_icon="🧠", layout="centered")

# --- Load model and tokenizer ---
@st.cache_resource
def load_all():
    model = load_model("model.h5")
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_all()
max_sequence_len = model.input_shape[1] + 1

# --- Sampling Function ---
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# --- App Header ---
st.markdown("""
# ✍️ LSTM Text Generator  
Give a few words, and let the AI complete your sentence creatively!  
Built with deep learning, streamlit, and a spark of curiosity 🚀
""")
st.divider()

# --- Theme Options ---
themes = {
    "Casual Chat 💬": "Hi, how are you",
    "Story Starter 📖": "Once upon a time",
    "Sci-Fi Tech 🤖": "In the year 2099",
    "Philosophical 💡": "The purpose of life is",
    "Text Message 📱": "Call me when you reach"
}

theme = st.selectbox("🎯 Choose a theme:", list(themes.keys()))
selected_theme_text = themes[theme]

# --- Random Prompt ---
prompts = [
    "He opened the door slowly and saw...",
    "Once upon a time in a world far away...",
    "The robot looked up and said...",
    "Love is not just a feeling, it's...",
    "Deep inside the cave, something moved...",
]

if st.button("🎲 Random Prompt"):
    st.session_state.random_text = random.choice(prompts)

# --- Input ---
default_text = st.session_state.get("random_text", selected_theme_text)
input_text = st.text_input("✏️ Enter your starting text:", value=default_text)

word_limit = st.slider("📝 Number of words to generate", 10, 200, 50, step=10)
temp_mode = st.radio("🌡️ Temperature mode (creativity)", ["Standard", "Surprise Me 😲"])

temperature = 0.8 if temp_mode == "Standard" else round(random.uniform(0.3, 1.4), 2)
st.caption(f"⚙️ Using temperature: `{temperature}`")

# --- Generate Text ---
if st.button("🚀 Generate Text"):
    if input_text.strip():
        text = input_text.strip()
        with st.spinner("Generating creative text..."):
            for _ in range(word_limit):
                token_list = tokenizer.texts_to_sequences([text])[0]
                token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
                preds = model.predict(token_list, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_word = next((word for word, index in tokenizer.word_index.items() if index == next_index), None)
                if not next_word:
                    break
                text += " " + next_word

        st.success(f"✅ Generated {word_limit} words!")
        st.markdown(f"### 📝 Output:\n**{text}**")
        st.download_button("💾 Download Text", text, file_name="generated_text.txt")
    else:
        st.warning("⚠️ Please enter some starting text.")

# --- Footer ---
st.divider()
st.markdown("""
Made with ❤️ using LSTM, Keras, and Streamlit 
""")
