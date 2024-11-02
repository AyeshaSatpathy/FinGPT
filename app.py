# Import os to set API key
import os
# Import necessary components from LangChain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
# Bring in Streamlit for UI/app interface
import streamlit as st
# Import PDF document loaders
from langchain.document_loaders import PyPDFLoader
# Import vector store and agent toolkit
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set API key for OpenAI Service (use environment variable for security)
# It's a good practice to set this in your environment instead of hardcoding it in the code
os.environ['OPENAI_API_KEY'] = 

# Create instance of OpenAI LLM and Embeddings
llm = OpenAI(temperature=0.0, verbose=True)
embeddings = OpenAIEmbeddings()

# Load PDF document
loader = PyPDFLoader('annualreport.pdf')
pages = loader.load_and_split()  # Split the PDF into individual pages

# Define a persistent directory to store the Chroma vectors
persist_directory = "./chroma_storage"  # Adjust this to a suitable directory path

# Load documents into vector store (Chroma) with a persistent storage directory
store = Chroma.from_documents(
    documents=pages, 
    embedding=embeddings, 
    collection_name='annualreport',
    persist_directory=persist_directory
)

# Create vectorstore info object - contains metadata about the vector store
vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="A banking annual report in PDF format",
    vectorstore=store
)

# Convert the document store into a LangChain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Create an agent executor using the LLM and the toolkit
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# Streamlit app title
st.title('🦜🔗 GPT Investment Banker')

# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user provides a prompt
if prompt:
    try:
        # Pass the prompt to the agent executor and display the response
        response = agent_executor.run(prompt)
        st.write(response)

        # With a Streamlit expander for Document Similarity Search
        with st.expander('Document Similarity Search'):
            # Use the prompt as the query to search for similar documents/pages
            search = store.similarity_search_with_score(prompt)
            
            if search:
                # Display the content of the first similar page
                st.write(search[0][0].page_content)
            else:
                st.write("No relevant pages found.")
    except Exception as e:
        if 'quota' in str(e).lower():
            st.error("Quota limit reached. Please try again later.")
        else:
            st.error("An error occurred: " + str(e))