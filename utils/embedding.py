import os
from glob import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone



class Embedding:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self._documents = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _load_documents(self, data_path, file_type, loader_class):
        """Helper function to load documents of a specific file type."""
        files = glob(f"{data_path}/*.{file_type}")
        for file in files:
            loader = loader_class(file)
            self._documents.extend(loader.load())
        return self._documents

    def load_data(self, data_path="data"):
        """Load text and PDF files from the specified data path."""
        self._load_documents(data_path, "txt", TextLoader)
        self._load_documents(data_path, "pdf", PyPDFLoader)
        return self._documents

    def get_embedding_model(self, model="nvidia/llama-3.2-nv-embedqa-1b-v1"):
        """Retrieve the NVIDIA embedding model."""
        return NVIDIAEmbeddings(model=model)

    def store_vectors(self, index_name):
        """Store document vectors in Pinecone."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            splits = text_splitter.split_documents(self._documents)
            PineconeVectorStore.from_documents(
                documents=splits,
                embedding=self.get_embedding_model(),
                index_name=index_name,
            )
        except Exception as e:
            print(f"Error storing vectors: {e}")
            return False
        return True

    def get_embedding(self, text):
        """Get embeddings for a given query text."""
        model = self.get_embedding_model()
        return model.embed_query(text)

    def get_retriever(self, index_name):
        """Initialize a retriever for a given index name."""
        vector_store = Pinecone.from_existing_index(
            index_name=index_name, embedding=self.get_embedding_model()
        )
        return vector_store.as_retriever(search_kwargs={"k": 10})


if __name__ == "__main__":
    load_dotenv()
    embedding = Embedding()
    embedding.load_data("../data")
    retriever = embedding.get_retriever(index_name="ustc-rag-2048")
    # os.environ['NVIDIA_API_KEY']
    # embedding.store_vectors(index_name="ustc-rag-2048")
