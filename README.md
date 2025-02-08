## Chatbot API with FAISS Embeddings

The demand for intelligent chatbot systems is growing across industries, from customer support and education to research and personal assistants. However, retrieving meaningful responses requires efficient text processing and embedding techniques. This project presents a chatbot API leveraging FAISS embeddings and Hugging Face models to provide relevant responses efficiently.

The goal of this project is to explore embedding-based retrieval systems, focusing on how FAISS indexing can improve chatbot interactions. By utilizing a fine-tuned deep learning model, we aim to deliver accurate and context-aware responses.

**We have gained insights from various research papers and frameworks mentioned below:**

### Technologies Used
- FAISS (Facebook AI Similarity Search) for efficient similarity-based search
- Hugging Face `HuggingFaceH4/zephyr-7b-alpha` model for text generation
- Flask & Flask-RESTful for API development
- LangChain for streamlined document processing and embedding
- Python 3.8+

#### BibTeX

```bibtex
@InProceedings{faiss2017,
    author = {Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
    title = {Billion-scale similarity search with GPUs},
    booktitle = {IEEE Transactions on Big Data},
    year = {2017}
```

### Dependencies

-Python 3
-FAISS
-Flask
-LangChain
-Hugging Face Transformers
### Test API

```
git clone https://github.com/your-username/chatbot-faiss
cd chatbot-faiss
pip install flask flask-restful transformers langchain faiss-cpu
python Store_embedding.py
python chatbot_api.py
curl -X POST "http://127.0.0.1:5000/chat" -H "Content-Type: application/json" -d '{"query": "What is AI?"}'
```

### FAISS-based Retrieval System

-Extract text from documents using `Data_extraction.py`.

-Convert text to vector embeddings using `Store_embedding.py`.

-Query embeddings using FAISS and generate responses using a Hugging Face model.

-Serve responses via a Flask-based API (`chatbot_api.py`).

### Qualitative Results

Performance is evaluated based on retrieval accuracy and response coherence.

### Ablation Study

The impact of different components like embedding models and FAISS parameters is studied to optimize chatbot response quality.

### License

This project is open-source and available under the MIT License.

