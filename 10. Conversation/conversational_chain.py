# This program is intended to demo the use of the following:
# - Extracts the contents of a webpage, chunks and loads the chunks (documents) into a FAISS db
# - Creates a sample conversation, uses a "create_history_aware_retriever" retrieval chain
# - The last step is to pass the history and ask a follow up question

from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os

loader = WebBaseLoader("https://www.wikihow.com/Do-Yoga")

docs = loader.load()

# The RecuresiveCharacterTextSplitter takes a large text and split it based on a specified chunk size.
# It does this by using a set of characters. The default characters provided to it are ["\n\n", "\n", " ", ""].
text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)

llm = GoogleGenerativeAI(
  model="gemini-1.5-pro-latest",
  temperature=0.5,
  google_api_key=os.getenv("GOOGLE_API_KEY"), 
)
embeddings = GoogleGenerativeAIEmbeddings(
  model="models/embedding-001",
  google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# FAISS (Facebook AI Similarity Search) is a library that allows developers to store and search for embeddings of 
# documents that are similar to each other. 
vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()

# First we need a prompt that we can pass into an LLM to generate this search query

prompt = ChatPromptTemplate.from_messages([
  MessagesPlaceholder(variable_name="chat_history"),
  ("user", "{input}"),
  ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
# If there is no chat_history, then the input is just passed directly to the retriever. If there is chat_history, then the prompt and LLM will be used to generate a search query, that search query is then passed to the retriever.
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

sample_answer = """Some key points for Yoga beginnerss are:
1. Find a comfortable place and time to practice.
2. Set a routine that suits you.
and so on...
"""

chat_history = [HumanMessage(content="What are the key things to consider for someone starting to practice Yoga?"), AIMessage(content=sample_answer)]

prompt = ChatPromptTemplate.from_messages([
  ("system", "Answer the user's questions based on the below context:\n\n{context}"),
  MessagesPlaceholder(variable_name="chat_history"),
  ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

output1 = retrieval_chain.invoke({
  "chat_history": chat_history,
  "input": "Can you elaborate on the first point?"
})

print(output1["answer"])

chat_history.extend([HumanMessage(content="Can you elaborate on the first point?"), AIMessage(content=output1["answer"])])

output2 = retrieval_chain.invoke({
  "chat_history": chat_history,
  "input": "What is the importance of finding a comfortable place to practice?"
})

print(output2["answer"])