from langchain_huggingface import HuggingFaceEmbeddings  # Corrected Import
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the extracted text from file
with open("text_data.txt", "r", encoding="utf-8") as file:
    text_data = file.read()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_text(text_data)

# Use Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings in FAISS
vector_store = FAISS.from_texts(texts, embeddings)
vector_store.save_local("faiss_index")

print("Embeddings stored successfully!")
