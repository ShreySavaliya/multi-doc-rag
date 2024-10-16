import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from logger import setup_logger

logging = setup_logger()

DOCUMENTS_DIR = "uploaded_documents"
PERSIST_DIRECTORY = "./chroma_db"
ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.csv', '.txt']

def ensure_documents_directory():
    """Ensure the documents directory exists."""
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    logging.info(f"Documents directory created: {DOCUMENTS_DIR}")

def get_document_list():
    """Return a list of documents in the documents directory."""
    ensure_documents_directory()
    return [f for f in os.listdir(DOCUMENTS_DIR) if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS]

def delete_document(filename):
    """Delete a document from the documents directory."""
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        delete_document_vectors(filename)
        logging.info(f"Deleted document: {file_path}")
        return True
    return False

def delete_document_vectors(filename):
    """Delete the vectors associated with a specific document."""
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="vector_db"
    )
    
    db.delete(where={"source": filename})
    logging.info(f"Deleted vectors for document: {filename}")

def display_documents():
    """Display the list of documents and provide delete functionality."""
    documents = get_document_list()
    
    if not documents:
        st.info("No documents found. Please upload some documents first.")
        logging.info("No documents found.")
        return

    for doc in documents:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(doc)
        with col2:
            if st.button(f"Delete", key=f"delete_{doc}"):   
                if delete_document(doc):
                    st.success(f"Deleted {doc}")
                    st.rerun()
                else:
                    st.error(f"Failed to delete {doc}")

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to the documents directory."""
    ensure_documents_directory()
    file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logging.info(f"File saved: {file_path}")
    return file_path

def get_file_extension(filename):
    """Get the file extension of a given filename."""
    return os.path.splitext(filename)[1].lower()