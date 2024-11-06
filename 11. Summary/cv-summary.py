from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import tempfile
import streamlit as st

load_dotenv()

def save_uploaded_file(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
        # Write the uploaded file's contents to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def process_docx(file_path):
    loader = Docx2txtLoader(file_path)
    text = loader.load_and_split()
    return text

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    text = ""
    for page in pages:
        text += page.page_content
    text = text.replace('\t', ' ')

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=50
    )
    texts = text_splitter.create_documents([text])
    return texts

def main():
    st.title("CV Summary Generator")

    uploaded_file = st.file_uploader("Select CV", type=["docx", "pdf"])

    text = ""
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]

        st.write("File Details:")
        st.write(f"File Name: {uploaded_file.name}")
        st.write(f"File Type: {file_extension}")

        # Save the uploaded file to a temporary location
        temp_file_path = save_uploaded_file(uploaded_file)

        try:
            if file_extension == "docx":
                text = process_docx(temp_file_path)
            elif file_extension == "pdf":
                text = process_pdf(temp_file_path)
            else:
                st.error("Unsupported file format. Please upload a .docx or .pdf file.")
                return

            llm = GoogleGenerativeAI(
                model="gemini-1.5-pro-latest",
                temperature=0.5,
                google_api_key=os.getenv("GOOGLE_API_KEY"), 
            )
            
            prompt_template = """You have been given a Resume to analyse. 
            Write a verbose detail of the following: 
            {text}
            Details:"""
            prompt = PromptTemplate.from_template(prompt_template)

            refine_template = (
                "Your job is to produce a final outcome\n"
                "We have provided an existing detail: {existing_answer}\n"
                "We want a refined version of the existing detail based on initial details below\n"
                "------------\n"
                "{text}\n"
                "------------\n"
                "Given the new context, refine the original summary in the following manner:"
                "Name: \n"
                "Email: \n"
                "Key Skills: \n"
                "Last Company: \n"
                "Experience Summary: \n"
            )
            refine_prompt = PromptTemplate.from_template(refine_template)
            
            chain = load_summarize_chain(
                llm=llm,
                chain_type="refine",
                question_prompt=prompt,
                refine_prompt=refine_prompt,
                return_intermediate_steps=True,
                input_key="input_documents",
                output_key="output_text",
            )
            
            result = chain({"input_documents": text}, return_only_outputs=True)
            st.write("Resume Summary:")
            st.text_area("Text", result['output_text'], height=400)

        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass

if __name__ == "__main__":
    main()