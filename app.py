from data.employees import generate_employee_data
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import logging
from assistant import Assistant
from prompts import SYSTEM_PROMPT, WELCOME_MESSAGE
from langchain_groq import ChatGroq
from gui import AssistantGUI
from langchain_community.embeddings import HuggingFaceEmbeddings


if __name__ == "__main__":

    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    st.set_page_config(page_title="Medlife Onboarding", layout="wide")

    @st.cache_data(ttl=3600, show_spinner="Loading Employee Data")
    def get_employee_data():
        return generate_employee_data(1)[0]
    
    @st.cache_resource(ttl=3600, show_spinner="Loading Vector Store...")
    def init_vector_store(pdf_path):
        try:
            pdf_loader = PyPDFLoader(pdf_path)
            documents = pdf_loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )

            text_chunks_docs = text_splitter.split_documents(documents)

            #embedding_model = OpenAIEmbeddings()
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            persistent_path = "data/vectorstore"

            vectorstore = Chroma.from_documents(
                documents=text_chunks_docs,
                embedding = embedding_model,
                persist_directory=persistent_path
            )

            return vectorstore
        except Exception as e:
            logging.error(f"Error in initializing vector store: {str(e)}")
            st.error(f"Failed to initialize vector store: {str(e)}")
            return None

    user_data = get_employee_data()

    vector_store = init_vector_store("data/medlifepolicy.pdf")

    if "customer" not in st.session_state:
        st.session_state.customer = user_data
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": WELCOME_MESSAGE}]

    llm=ChatGroq(
    model="llama-3.3-70b-versatile"
    )

    assistant = Assistant(
        system_prompt=SYSTEM_PROMPT,
        llm=llm,
        message_history=st.session_state.messages,
        employee_information=st.session_state.customer,
        vector_store=vector_store
    )

    gui = AssistantGUI(assistant)
    gui.render()


    


