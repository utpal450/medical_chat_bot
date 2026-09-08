from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document


# Extract Data From PDF File
def load_pdf_documents(pdf_directory):
    """
    Load all PDF files from the specified directory.
    """

    loader = DirectoryLoader(
        pdf_directory,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    return documents


# Split documents into chunks
def text_split(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    texts_chunks = text_splitter.split_documents(documents)

    return texts_chunks


# Generate embeddings
def generate_embeddings(
    texts_chunks: List[Document]
) -> List[List[float]]:

    texts = [
        doc.page_content
        for doc in texts_chunks
    ]

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_embeddings = embeddings.embed_documents(
        texts
    )

    return vector_embeddings
