from flask import Flask, render_template, request, session, Response
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompts import system_template
import os


app = Flask(__name__)

load_dotenv()

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "medical-chatbot-secret-key"
)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -----------------------------------
# Pinecone
# -----------------------------------

index_name = "medical-chatbot"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# -----------------------------------
# LLM
# -----------------------------------

chatModel = ChatOpenAI(
    model="gpt-4o",
    streaming=True
)


# -----------------------------------
# Prompt
# -----------------------------------




prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("human", "{input}"),
    ]
)


# -----------------------------------
# RAG Chain
# -----------------------------------

question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# -----------------------------------
# Home
# -----------------------------------

@app.route("/")
def index():

    if "chat_history" not in session:
        session["chat_history"] = []

    return render_template("chat.html")


# -----------------------------------
# Streaming Chat
# -----------------------------------

@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form["msg"]

    print("Question:", msg)

    chat_history = session.get("chat_history", [])

    history_text = ""

    for message in chat_history:
        history_text += f"User: {message['user']}\n"
        history_text += f"Assistant: {message['assistant']}\n"


    def generate():

        full_answer = ""

        for chunk in rag_chain.stream({
            "input": msg,
            "chat_history": history_text
        }):

            answer_chunk = chunk.get("answer", "")

            if answer_chunk:

                full_answer += answer_chunk

                yield answer_chunk


        # Save complete answer to memory
        chat_history.append({
            "user": msg,
            "assistant": full_answer
        })

        session["chat_history"] = chat_history

        print("Response:", full_answer)


    return Response(
        generate(),
        content_type="text/plain; charset=utf-8"
    )


# -----------------------------------
# Clear Chat
# -----------------------------------

@app.route("/clear", methods=["POST"])
def clear_chat():

    session["chat_history"] = []

    return {
        "message": "Chat history cleared"
    }


# -----------------------------------
# Run Flask
# -----------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )