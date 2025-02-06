from langchain_community.document_loaders import UnstructuredURLLoader
import sys
sys.stdout.reconfigure(encoding='utf-8')

urls = ["https://brainlox.com/courses/category/technical"]
loader = UnstructuredURLLoader(urls=urls)
documents = loader.load()
text_data = "\n".join([doc.page_content for doc in documents])
print(text_data)  # Verify extracted content

