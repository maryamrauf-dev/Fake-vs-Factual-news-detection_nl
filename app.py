import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# App title
# -----------------------------
st.title("Fake vs Factual News Classification")
st.write("""
This app classifies news text as Fake or Factual.
Type your own text, upload CSVs, and see model evaluation visuals.
""")

# -----------------------------
# Sidebar input for single prediction
# -----------------------------
user_input = st.sidebar.text_area("Enter your news text here:")

# -----------------------------
# Load saved models and vectorizer
# -----------------------------
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("countvectorizer.pkl")

# Optional: Load test data for evaluation visuals
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

# -----------------------------
# Single input prediction
# -----------------------------
if st.sidebar.button("Predict"):
    vect_text = vectorizer.transform([user_input])
    prediction = model.predict(vect_text)
    st.success(f"Predicted Class: {prediction[0]}")

# -----------------------------
# Batch prediction via CSV
# -----------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'text' not in data.columns:
        st.error("CSV must have a column named 'text'")
    else:
        vect_data = vectorizer.transform(data['text'])
        data['Prediction'] = model.predict(vect_data)
        st.dataframe(data)

# -----------------------------
# Model evaluation visuals
# -----------------------------
st.header("Model Evaluation")
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
st.write(pd.DataFrame(report).transpose())
