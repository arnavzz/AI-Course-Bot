import os
import time
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Load environment variables (for API key)
groq_api_key = os.getenv("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")
if not groq_api_key:
    raise ValueError("API Key is missing. Please make sure the .env file contains the correct key.")

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Adjusted for HuggingFace embeddings
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")  # Replacing with ChatGroq model

# Create prompt template
prompt = """
    You are a friendly AI assistant. Your goal is to provide helpful and engaging responses to user inquiries.
    If the information is in the context, use it directly and respond conversationally.
    If the information is not in the context, provide a direct, helpful answer.
    <context>
    {context}
    </context>
    Question: {input}
    Answer:
"""

def create_vector_embedding():
    try:
        # Load and process the text file
        loader = TextLoader("scalixity_faq_data.txt")  # Load the FAQ file for embedding
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Create FAISS vector store from documents
        vectorstore = FAISS.from_documents(texts, embeddings)

        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None

# Flask app initialization
app = Flask(__name__)
api = Api(app)

# Create ConversationalRetrievalChain
def create_retrieval_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, chain_type="stuff"
    )
    return qa_chain

class Chatbot(Resource):
    def post(self):
        try:
            # Parse incoming request
            data = request.get_json()
            if not data or "query" not in data:
                return jsonify({"error": "Invalid JSON: 'query' key is required"}), 400
            
            query = data["query"]

            # Create vectorstore and retrieval chain
            vectorstore = create_vector_embedding()
            if not vectorstore:
                return jsonify({"error": "Failed to initialize the knowledge base."}), 500

            qa_chain = create_retrieval_chain(vectorstore)

            # Process the query using the retrieval chain
            response = qa_chain.invoke({"question": query, "chat_history": []})
            return jsonify({"response": response['answer']})

        except Exception as e:
            return jsonify({"error": f"Error processing question: {str(e)}"}), 500

api.add_resource(Chatbot, "/chat")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
