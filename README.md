
# 📩 Message Predictor

A machine learning-based application that predicts the intent or category of a given message using natural language processing (NLP). This project is aimed at enhancing automated text understanding for applications like smart replies, spam detection, chatbot assistance, and more.

---

## 🚀 Features

- 📊 Text preprocessing with NLTK / spaCy
- 🧠 ML/DL models for classification (Logistic Regression, SVM, or Transformers)
- 🗃️ Support for custom datasets (CSV/Text)
- 📈 Model training and evaluation pipeline
- 🖥️ GUI or CLI for inputting and predicting messages
- 📁 Modular and scalable codebase

---

## 🧠 Model Architecture

> Describe your model(s) here. For example:

- **TF-IDF + Logistic Regression**
- **CountVectorizer + SVM**
- **BERT or DistilBERT for sentence embeddings**
- Performance evaluation with accuracy, precision, recall, F1-score

---

### 1. Clone the Repository

```bash
git clone https://github.com/AnuragAgrahari04/Message_Predictor.git
cd Message_Predictor
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install the Required Dependencies

```bash
pip install -r requirements.txt
```

### 4. Streamlit Run

```bash
streamlit run app.py
```


## 📂 Project Structure

```bash
Message_Predictor/
├── data/               # Dataset folder (CSV/Text files)
├── models/             # Saved models (pickle or h5)
├── utils/              # Helper functions (preprocessing, etc.)
├── main.py             # Main execution script
├── train.py            # Training pipeline
├── predict.py          # Prediction script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```


## 📈 Evaluation

| Metric       | Score |
|--------------|-------|
| Accuracy     | 0.93  |
| Precision    | 0.91  |
| Recall       | 0.92  |
| F1 Score     | 0.91  |





