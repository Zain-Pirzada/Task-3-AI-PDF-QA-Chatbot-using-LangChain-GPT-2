#%%writefile app.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import os

# Set environment variable for Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Streamlit UI
st.title('PDF QA Chatbot')
st.write('Upload a PDF file and ask questions about its content.')

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Read text from PDF
    pdfreader = PdfReader(uploaded_file)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    # Display raw text (optional)
    st.write("Extracted Text:")
    st.write(raw_text)

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create FAISS index
    document_search = FAISS.from_texts(texts, embeddings)

    # Load QA chain
    llm = HuggingFaceHub(repo_id="openai-community/gpt2", model_kwargs={"temperature":5, "max_length":512})
    chain = load_qa_chain(llm, chain_type="stuff")

    # User input for query
    query = st.text_input("Ask a question about the PDF content")

    if query:
        # Perform similarity search and run the QA chain
        docs = document_search.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)

        # Display the answer
        st.write("Answer:")
        st.write(answer)


from pyngrok import ngrok

# Terminate any existing tunnels
ngrok.kill()

# Start the Streamlit app in the background
#!streamlit run app.py &

# Create a public URL for the Streamlit app
public_url = ngrok.connect(port='8501')
print(f'Public URL: {public_url}')
