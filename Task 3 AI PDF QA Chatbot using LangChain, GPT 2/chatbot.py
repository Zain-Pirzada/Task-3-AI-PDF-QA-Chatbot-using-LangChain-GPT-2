from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


# provide the path of  pdf file/files.
pdfreader = PdfReader('/content/TAC L1 TroubleShooting Guide v1.22.pdf')

from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content


#raw_text

# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 600,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


#len(texts)


from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()


document_search = FAISS.from_texts(texts, embeddings)

#document_search


from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub

llm=HuggingFaceHub(repo_id="openai-community/gpt2", model_kwargs={"temperature":5, "max_length":512})

chain = load_qa_chain(llm, chain_type="stuff")

query = "Digital Box Channels Missing, SD Channels"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)

