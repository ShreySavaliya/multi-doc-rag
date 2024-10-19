import os
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredFileLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from documents import DOCUMENTS_DIR, get_file_extension
from logger import setup_logger

logging = setup_logger()

class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        persist_directory: str = "./chroma_db",
        collection_name: str = "vector_db",
    ):
        """
        Initializes the EmbeddingsManager with the specified model and Chroma settings.

        Args:
            model_name (str): The HuggingFace model name for embeddings.
            device (str): The device to run the model on ('cpu' or 'cuda').
            encode_kwargs (dict): Additional keyword arguments for encoding.
            persist_directory (str): The directory to persist the Chroma database.
            collection_name (str): The name of the Chroma collection.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

    def process_pdf(self, pdf_path: str):
        """
        Processes a single PDF and returns the text splits.
        """
        if not os.path.exists(pdf_path):
            logging.info(f"The file {pdf_path} does not exist.")
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")

        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            logging.error(f"No documents were loaded from {pdf_path}.")
            raise ValueError(f"No documents were loaded from {pdf_path}.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250
        )
        splits = text_splitter.split_documents(docs)
        if not splits:
            logging.error(f"No text chunks were created from {pdf_path}.")
            raise ValueError(f"No text chunks were created from {pdf_path}.")

        return splits

    # def create_embeddings_for_multiple_pdfs(self, pdf_paths: list):
    #     """
    #     Processes multiple PDFs, creates embeddings, and stores them in Chroma.

    #     Args:
    #         pdf_paths (list): List of file paths to PDF documents.

    #     Returns:
    #         str: Success message upon completion.
    #     """
    #     all_splits = []
    #     for pdf_path in pdf_paths:
    #         try:
    #             splits = self.process_pdf(pdf_path)
    #             all_splits.extend(splits)
    #         except Exception as e:
    #             logging.error(f"Failed to process PDF {pdf_path}: {e}")

    #     if not all_splits:
    #         logging.error("No valid text chunks were created from any of the PDFs.")
    #         raise ValueError("No valid text chunks were created from any of the PDFs.")

    #     # Create and store embeddings in Chroma
    #     try:
    #         chroma = Chroma(
    #             persist_directory=self.persist_directory,
    #             embedding_function=self.embeddings,
    #             collection_name=self.collection_name
    #         )

    #         batch_size = 100 
    #         for i in range(0, len(all_splits), batch_size):
    #             batch = all_splits[i:i+batch_size]
    #             chroma.add_documents(documents=batch)

    #     except Exception as e:
    #         logging.error(f"Failed to create or store embeddings in Chroma: {e}")
    #         raise ConnectionError(f"Failed to create or store embeddings in Chroma: {e}")

    #     logging.info(f"Vector DB Successfully Created and Stored in Chroma for {len(pdf_paths)} PDFs!")
    #     return f"✅ Vector DB Successfully Created and Stored in Chroma for {len(pdf_paths)} PDFs!"


    def process_document(self, file_path: str):
        """
        Processes a single document and returns the text splits with page information.
        """
        if not os.path.exists(file_path):
            logging.info(f"The file {file_path} does not exist.")
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        file_extension = get_file_extension(file_path)
        
        if file_extension == '.pdf':
            loader = UnstructuredPDFLoader(file_path)
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)

        docs = loader.load()
        if not docs:
            logging.error(f"No documents were loaded from {file_path}.")
            raise ValueError(f"No documents were loaded from {file_path}.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250
        )
        splits = text_splitter.split_documents(docs)
        if not splits:
            logging.error(f"No text chunks were created from {file_path}.")
            raise ValueError(f"No text chunks were created from {file_path}.")

        # Add page information to metadata
        # for i, split in enumerate(splits):
        #     split.metadata['page'] = i + 1  # Assuming each split is a page
        #     split.metadata['source'] = os.path.basename(file_path)

        return splits
    
    def create_embeddings_for_multiple_documents(self, file_paths: list):
        """
        Processes multiple documents, creates embeddings, and stores them in Chroma.

        Args:
            file_paths (list): List of file paths to documents.

        Returns:
            str: Success message upon completion.
        """
        all_splits = []
        for file_path in file_paths:
            try:
                splits = self.process_document(file_path)
                all_splits.extend(splits)
            except Exception as e:
                logging.error(f"Failed to process document {file_path}: {e}")

        if not all_splits:
            logging.error("No valid text chunks were created from any of the documents.")
            raise ValueError("No valid text chunks were created from any of the documents.")

        # Create and store embeddings in Chroma
        try:
            chroma = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )

            batch_size = 100 
            for i in range(0, len(all_splits), batch_size):
                batch = all_splits[i:i+batch_size]
                chroma.add_documents(documents=batch)

        except Exception as e:
            logging.error(f"Failed to create or store embeddings in Chroma: {e}")
            raise ConnectionError(f"Failed to create or store embeddings in Chroma: {e}")

        logging.info(f"Vector DB Successfully Created and Stored in Chroma for {len(file_paths)} documents!")
        return f"✅ Vector DB Successfully Created and Stored in Chroma for {len(file_paths)} documents!"

    
if __name__ == "__main__":
    embeddings_manager = EmbeddingsManager()
    pdf_path = "temp.pdf"
    result = embeddings_manager.create_embeddings(pdf_path)
    print(result)