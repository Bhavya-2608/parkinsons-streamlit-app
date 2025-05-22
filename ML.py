import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc

st.set_page_config(page_title="Parkinson's Disease Detection App", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("parkinsons.csv")
    X = df.drop(['name', 'status'], axis=1)
    y = df['status']
    return df, X, y

# Train Model
@st.cache_data
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    return model, scaler, X_scaled

df, X, y = load_data()
model, scaler, X_scaled = train_model(X, y)

# App Title
st.title("🧠 Parkinson's Disease Detection App")
st.markdown("This app uses voice measurements to detect whether a person has **Parkinson's Disease**.")

# Sidebar Inputs
st.sidebar.header("🔧 Input Options")

input_mode = st.sidebar.radio("Choose input method", ["Manual Input", "Use Random Sample"])

def get_input_data():
    if input_mode == "Manual Input":
        features = {}
        for col in X.columns:
            step = 0.01 if df[col].dtype != 'int64' else 1
            features[col] = st.sidebar.slider(
                label=col,
                min_value=float(df[col].min()),
                max_value=float(df[col].max()),
                value=float(df[col].mean()),
                step=step
            )
        return pd.DataFrame([features]), None
    else:
        selected_index = st.sidebar.slider("Select sample index", 0, len(df) - 1, 0)
        return X.iloc[[selected_index]], selected_index

input_df, selected_index = get_input_data()

# Show selected sample name if random sample is used
if input_mode == "Use Random Sample" and selected_index is not None:
    st.write(f"Selected Sample Name: **{df.loc[selected_index, 'name']}**")

input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Main Output
st.subheader('🧾 Prediction Result')
if prediction[0] == 0:
    st.success("🟢 No Parkinson's Detected")
else:
    st.error("🔴 Parkinson's Disease Detected")

st.write(f"**Probability:** No Disease: `{prediction_proba[0][0]:.2f}`, Disease: `{prediction_proba[0][1]:.2f}`")

# Optional Download
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

result_data = input_df.copy()
result_data['Prediction'] = prediction
csv = convert_df(result_data)

st.download_button(
    label="📥 Download Prediction Result",
    data=csv,
    file_name='parkinsons_prediction.csv',
    mime='text/csv',
)

# Dataset Insights with expanders for cleaner UI
st.subheader("📊 Dataset Insights")

with st.expander("📊 Model Accuracy & ROC Curve"):
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    st.write(f"Accuracy on training data: **{accuracy:.2f}**")

    fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)

with st.expander("🔍 Dataset Preview"):
    st.dataframe(df.head())

with st.expander("📈 Class Distribution"):
    class_counts = df['status'].value_counts().rename({0: "No Disease", 1: "Disease"})
    st.bar_chart(class_counts)

with st.expander("📉 Statistical Summary"):
    st.dataframe(df.describe())

with st.expander("🔗 Feature Correlation"):
    st.dataframe(df.corr(numeric_only=True))

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend()
    st.pyplot(fig)
