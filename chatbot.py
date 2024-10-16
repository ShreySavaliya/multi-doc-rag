import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import streamlit as st
import config
from logger import setup_logger

logging = setup_logger()

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "mixtral-8x7b-32768",
        llm_temperature: float = 0.3,
        persist_directory: str = "./chroma_db",
        collection_name: str = "vector_db",
    ):
        """
        Initializes the ChatbotManager with embedding models, LLM, and vector store.

        Args:
            model_name (str): The HuggingFace model name for embeddings.
            device (str): The device to run the model on ('cpu' or 'cuda').
            encode_kwargs (dict): Additional keyword arguments for encoding.
            llm_model (str): The local LLM model name for ChatOllama.
            llm_temperature (float): Temperature setting for the LLM.
            persist_directory (str): The directory to persist the Chroma database.
            collection_name (str): The name of the Chroma collection.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        self.llm = ChatGroq(
            model=self.llm_model, 
            temperature=self.llm_temperature, 
            max_tokens=2048, 
            groq_api_key=config.GROQ_API_KEY)

        self.prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer and be wise while answering the question. And don't just include the unnecessary details in the answer just to increase the length of the answer. Answer must be detailed and well explained.
"""
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        self.retriever = self.db.as_retriever(
            search_kwargs={"k": 5})

        self.chain_type_kwargs = {"prompt": self.prompt}

        # Initialize the RetrievalQA chain with return_source_documents=False
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,  # Set to False to return only 'result'
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False
        )

    def get_response(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """
        try:
            response = self.qa.invoke(query)
            return response['result'] 
        except Exception as e:
            st.error(f"⚠️ An error occurred while processing your request: {e}")
            logging.info(f"An error occurred while processing the request: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."


if __name__ == "__main__":
    chatbot = ChatbotManager()
    user_input = "What is the capital of France?"
    response = chatbot.get_response(user_input)
    print(response)