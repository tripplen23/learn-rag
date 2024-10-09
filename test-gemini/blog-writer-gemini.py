# This program is intended to Write Blogs by referring to given webpage:
# This uses Googlge Gemini

# Load the content from the given URL -> Split the text from document -> Embedding those Splitted text into vector -> Put it into a vector database

from langchain_community.chat_models import ChatOpenAI
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

loader = WebBaseLoader("https://medium.com/swlh/algorithmic-management-what-is-it-and-whats-next-33ad3429330b")

docs = loader.load()

# The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size.
# It does this by using a set of characters. The default characters provided to it are ["\n\n", "\n", " ", ""]
text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)

#llm = ChatOpenAI(model_name="gpt-3.5-turbo")
llm =  ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# FAISS ()