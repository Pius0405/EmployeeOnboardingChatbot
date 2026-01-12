# Employee Onboarding Chatbot 🤖

## Overview
The **Employee Onboarding Chatbot** is an AI-powered assistant designed to help new employees quickly understand company policies, procedures, and internal documentation. Built using **LangChain** and modern LLM integrations, the chatbot leverages **Retrieval-Augmented Generation (RAG)** to provide accurate, document-grounded answers instead of generic responses.

This project is intended as a functional prototype for employee onboarding, HR assistance, and internal knowledge management.

---

## Key Features
- 📄 Document-based Question Answering using company policy PDFs  
- 🧠 Retrieval-Augmented Generation (RAG) with vector embeddings  
- ⚡ Fast semantic search using Chroma vector store  
- 🔐 Secure API key handling with environment variables  
- 🖥️ Interactive UI powered by Streamlit  
- 🧪 Synthetic data generation support using Faker  

---

## Tech Stack

### Core Frameworks
- **LangChain** – orchestration of LLMs and retrieval pipelines  
- **Streamlit** – lightweight and interactive frontend UI  

### LLM Providers
- **OpenAI** (via `langchain-openai`)  
- **Groq** (via `langchain-groq`)  

### Vector Store & Embeddings
- **Chroma** – local vector database  
- **Sentence Transformers** – document embeddings  

### Document Processing
- **PyPDF** – PDF parsing and ingestion  

---

## Project Structure
```text
EmployeeOnboardingChatbot/
│
├── app.py              # Main application entry point
├── assistant.py        # Core chatbot and RAG logic
├── gui.py              # Streamlit UI components
├── prompts.py          # Prompt templates and system messages
├── data/               # Document storage & vector indexes (ignored in git)
├── .env                # API keys and secrets (ignored)
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
