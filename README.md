# VitaeRAG: Local Knowledge Base Q&A with RAG


This is a local knowledge base Q&A project based on **Retrieval-Augmented Generation (RAG)** technology.  
You can upload your own PDF papers, and the project will process them into a vector database.  
Then, through a command-line interface, you can query the contents of these papers using a Large Language Model (LLM).

## 📂 Project Structure

```tree
rag-llm-project/
├── data/
│   └── (Please put your PDF papers here)
├── vector_store/
│   └── (This folder will be automatically created by ingest.py to store vector indexes)
├── ingest.py          # 1. Script to process PDFs and create the vector database
├── app.py             # 2. Main application for Q&A with your papers
├── requirements.txt   # 3. Python dependencies required for the project
└── README.md          # 4. Project documentation
```

## 🚀 Usage Steps

### 1. Environment Setup
First, ensure you have **Python 3.8 or higher** installed.  
Then, clone or download this project, open a terminal in the project root directory, and (optionally) create a virtual environment before installing dependencies:

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
