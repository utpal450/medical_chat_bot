from dotenv import load_dotenv
import os
from src.helpers import load_pdf_file, filter_to_minimal_docs, text_split, generate_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


extracted_data=load_pdf_file(data='data/')


texts_chunks=text_split(extracted_data)

embeddings = generate_embeddings(texts_chunks)

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=len(embeddings[0]),
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    

index = pc.Index(index_name)



docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunks,
    index_name=index_name,
    embedding=embeddings, 
)