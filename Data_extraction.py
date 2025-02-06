from langchain_community.document_loaders import UnstructuredURLLoader

# Define the URL to extract data from
urls = ["https://brainlox.com/courses/category/technical"]
loader = UnstructuredURLLoader(urls=urls)

# Load and extract text data
documents = loader.load()
text_data = "\n".join([doc.page_content for doc in documents])

# Save the extracted text to a file
with open("text_data.txt", "w", encoding="utf-8") as file:
    file.write(text_data)

print("Data extraction complete. Saved to text_data.txt")
