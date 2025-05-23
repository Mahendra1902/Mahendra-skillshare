import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Constants
os.environ["GOOGLE_API_KEY"] = "AIzaSyBmUYQdImYbjPJesYFoMHVEfibp5l1CKBc"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# UI Setup
st.set_page_config(page_title="AI-Powered Safety Monitoring", layout="wide")
st.title("AI-Powered Industrial Safety Monitoring")

st.sidebar.header("Control Panel")
shift_start = st.sidebar.time_input("Shift Start Time", value=datetime.time(8, 0))
selected_area = st.sidebar.selectbox("Select Area", ["Furnace", "Boiler Room", "Assembly Line"])

# Real-time Alerts
st.header("Real-time Hazard Alerts")
col1, col2 = st.columns(2)

def analyze_video_feed():
    return ["No helmet detected near Furnace 3", "Unauthorized entry detected in Zone B"]

def check_sensor_data(sensor_df):
    alerts = []
    if sensor_df['gas_level'].iloc[-1] > 300:
        alerts.append("High gas level detected")
    if sensor_df['temperature'].iloc[-1] > 80:
        alerts.append("High temperature in Boiler Room")
    if sensor_df['noise_level'].iloc[-1] > 85:
        alerts.append("Noise level exceeds safety threshold")
    return alerts

def generate_prevention_checklist():
    return ["Wear helmet and safety gear", "Check gas detector calibration", "Inspect fire extinguishers"]

def generate_compliance_report():
    return "Safety compliance is at 92% this month. Helmet violations decreased by 15%."

with col1:
    st.subheader("Video Surveillance Alerts")
    for alert in analyze_video_feed():
        st.error(f"📹 {alert}")

with col2:
    st.subheader("Sensor Alerts")
    sensor_df = pd.DataFrame({
        'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=10, freq='min'),
        'gas_level': np.random.randint(250, 350, 10),
        'temperature': np.random.randint(60, 90, 10),
        'noise_level': np.random.randint(70, 95, 10)
    })
    for alert in check_sensor_data(sensor_df):
        st.error(f"📊 {alert}")

st.header("Prevention Checklist")
for item in generate_prevention_checklist():
    st.checkbox(item, value=False)

st.header("Historical Incident Analysis (Upload and Ask)")
uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader("temp_uploaded.pdf")
    else:
        with open("temp_uploaded.txt", "wb") as f:
            f.write(uploaded_file.read())
        loader = TextLoader("temp_uploaded.txt")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3),
        retriever=retriever,
        chain_type="stuff"
    )

    user_query = st.text_input("Ask about this document (e.g., helmet violation near furnace):")
    if user_query:
        with st.spinner("Searching uploaded content..."):
            result = qa_chain.run(user_query)
            st.info(result)

st.header("Safety Compliance Report")
st.success(generate_compliance_report())

st.header("Sensor Readings (Last 10 min)")
st.dataframe(sensor_df.set_index('timestamp'))

st.header("Add New Incident")
with st.form("incident_form"):
    incident_date = st.date_input("Incident Date")
    incident_desc = st.text_area("Incident Description")
    action_taken = st.text_area("Action Taken")

    if st.form_submit_button("Submit Incident"):
        new_incident = f"{incident_date}: {incident_desc} Action: {action_taken}"
        os.makedirs("incident_docs", exist_ok=True)
        with open(f"incident_docs/incident_{len(os.listdir('incident_docs'))+1}.txt", "w") as f:
            f.write(new_incident)
        st.success("Incident logged. It will be available when re-uploaded.")
