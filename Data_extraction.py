from langchain.document_loaders import UnstructuredURLLoader

urls = ["https://brainlox.com/courses/category/technical"]
loader = UnstructuredURLLoader(urls=urls)
documents = loader.load()
text_data = "\n".join([doc.page_content for doc in documents])
print(text_data)  # Verify extracted content

