from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

app = Flask(__name__)
api = Api(app)

# Load FAISS embeddings
vector_store = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True  # Add this flag to bypass security warning
)


# Load a local Hugging Face model for generating responses
hf_pipeline = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha")
llm = HuggingFacePipeline(pipeline=hf_pipeline)

class Chatbot(Resource):
    def post(self):
        query = request.json["query"]
        docs = vector_store.similarity_search(query, k=3)

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)

        return jsonify({"response": response})

api.add_resource(Chatbot, "/chat")

if __name__ == "__main__":
    app.run(debug=True)
