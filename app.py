'''
References: 
  https://github.com/nicknochnack/LangchainDocuments
  https://github.com/nicknochnack/Langchain-Crash-Course
'''

import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_vectorstore_agent, \
                                            VectorStoreToolkit, \
                                            VectorStoreInfo


persist_chroma_directory = '.chroma_db'

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YourAPIKey')

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.0, verbose=True)

# Create and load PDF Loader
#loader = PyPDFLoader('annualreport.pdf')
# Split pages from pdf 
#pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
#store = Chroma.from_documents(pages, collection_name='annualreport')

# use OpenAI embedding
embedding = OpenAIEmbeddings()
chroma_store = Chroma(persist_directory=persist_chroma_directory, embedding_function=embedding)

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="pdf_docs",
    description="policies as a pdf",
    vectorstore=chroma_store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

st.title('🦜🔗 GPT Airline Policies')
prompt = st.text_input('Input your prompt here')


if prompt:
    response = agent_executor.run(prompt)
    st.write(response)

    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = chroma_store.similarity_search_with_score(prompt) 
        # Write out the first 
        #st.write(search[0][0].page_content,
        #         search[0][0].metadata) 
        st.write(search)
        