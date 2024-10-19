import streamlit as st
from streamlit import session_state
import time
import base64
import os
from vectors import EmbeddingsManager 
from chatbot import ChatbotManager    
from documents import display_documents, save_uploaded_file, DOCUMENTS_DIR, ALLOWED_EXTENSIONS
from logger import setup_logger

logging = setup_logger()

if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Set the page configuration to wide layout and add a title
st.set_page_config(
    page_title="Chat-Book App",
    layout="wide",
    initial_sidebar_state="expanded",
)   

# Navigation Menu
menu = ["🏠 Home", "🤖 Chatbot", "📚 Documents"]
choice = st.selectbox("Navigate", menu)

# Home Page
if choice == "🏠 Home":
    st.title("📄 Chat-Book App")
    st.markdown("""
    Welcome to **Chat-Book App**! 🚀

    **Built using Open Source Stack (Mixtral, BGE Embeddings, and ChromaDB.)**

    - **Upload Documents**: Easily upload your PDF documents.
    - **Chat**: Interact with your documents through our intelligent chatbot.

    Enhance your document management experience with Chat-Book App! 😊
    """)

# Chatbot Page
elif choice == "🤖 Chatbot":
    st.title("🤖 Chatbot Interface")
    st.markdown("---")
    
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF files", type=ALLOWED_EXTENSIONS, accept_multiple_files=True)
    logging.info(f"Files Uploaded!")
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} File(s) Uploaded Successfully!")
        
        # Save the uploaded files to the documents directory
        temp_pdf_paths = []
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            temp_pdf_paths.append(file_path)
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**File Size:** {uploaded_file.size} bytes")
        
        # Store the temp_pdf_paths in session_state
        st.session_state['temp_pdf_paths'] = temp_pdf_paths

    st.markdown("---")

    st.header("🧠 Embeddings")
    create_embeddings = st.checkbox("✅ Create Embeddings")
    if create_embeddings:
        if not st.session_state.get('temp_pdf_paths'):
            st.warning("⚠️ Please upload PDF files first.")
        else:
            try:
                # Initialize the EmbeddingsManager
                embeddings_manager = EmbeddingsManager(
                    model_name="BAAI/bge-small-en",
                    device="cpu",
                    encode_kwargs={"normalize_embeddings": True},
                    persist_directory="./chroma_db",
                    collection_name="vector_db"
                )
                
                with st.spinner("🔄 Embeddings are in process..."):
                    result = embeddings_manager.create_embeddings_for_multiple_documents(st.session_state['temp_pdf_paths'])
                    time.sleep(1)
                logging.info(f"Embeddings created")
                st.success(result)
                
                # Initialize the ChatbotManager after embeddings are created
                if st.session_state['chatbot_manager'] is None:
                    st.session_state['chatbot_manager'] = ChatbotManager(
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        llm_model="mixtral-8x7b-32768",
                        llm_temperature=0.3,
                        persist_directory="./chroma_db",
                        collection_name="vector_db"
                    )
                
            except FileNotFoundError as fnf_error:
                st.error(fnf_error)
            except ValueError as val_error:
                st.error(val_error)
            except ConnectionError as conn_error:
                st.error(conn_error)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    st.markdown("---")

    st.header("💬 Chat with Document")
    
    if st.session_state['chatbot_manager'] is None:
        st.info("🤖 Please upload a PDF and create embeddings to start chatting.")
    else:
        # Display existing messages
        for msg in st.session_state['messages']:
            st.chat_message(msg['role']).markdown(msg['content'])

        # User input
        if user_input := st.chat_input("Type your message here..."):

            st.chat_message("user").markdown(user_input)
            st.session_state['messages'].append({"role": "user", "content": user_input})

            with st.spinner("🤖 Responding..."):
                try:
                    answer = st.session_state['chatbot_manager'].get_response(user_input)
                    time.sleep(1)

                    # Display the chatbot's response (answer and sources)
                    st.chat_message("assistant").markdown(f"**Answer:** {answer['response']}")  # Display the answer
                    if answer.get('sources'):  # Check if sources are available
                        st.chat_message("assistant").markdown(f"**Sources:** {answer['sources']}")

                except Exception as e:
                    answer = f"⚠️ An error occurred while processing your request: {e}"
            
            # st.chat_message("assistant").markdown(answer)
            st.session_state['messages'].append({"role": "assistant", "content": answer})

# Documents Page
elif choice == "📚 Documents":
    st.title("📚 Documents")
    display_documents()

    st.markdown("---")
    st.header("📤 Upload New Documents")
    new_uploads = st.file_uploader("Upload additional PDF files", type=ALLOWED_EXTENSIONS, accept_multiple_files=True)
    
    if new_uploads:
        for uploaded_file in new_uploads:
            file_path = save_uploaded_file(uploaded_file)
            st.success(f"Uploaded: {uploaded_file.name}")
