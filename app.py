import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# Set the model and tokenizer
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32, low_cpu_mem_usage=True)

# Function to load and preprocess the file
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)  # Increased chunk size
    texts = text_splitter.split_documents(pages)
    
    return [text.page_content for text in texts]

# LLM summarization pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline('summarization', model=base_model, tokenizer=tokenizer, max_length=500, min_length=50)
    input_texts = file_preprocessing(filepath)
    
    full_summary = ""

    for text_chunk in input_texts:
        try:
            result = pipe_sum(text_chunk)
            full_summary += result[0]['summary_text'] + "\n"
        except Exception as e:
            st.error(f"An error occurred during summarization: {str(e)}")
    
    return full_summary

@st.cache_data
# Function to display the PDF file
def displayPDF(file):
    with open(file, 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit main app logic
st.set_page_config(layout='wide')

def main():
    st.title('Document Summarization')
    
    uploaded_file = st.file_uploader('Upload your PDF', type=['pdf'])
    
    if uploaded_file is not None:
        if st.button('Summarize'):
            col1, col2 = st.columns(2)
            filepath='data/'+uploaded_file.name
            with open(filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info('Uploaded PDF file:')
                displayPDF(filepath)
            with col2:
                st.info('Summarization result:')
                summary = llm_pipeline(filepath)
                st.success(summary)

if __name__ == '__main__':
    main()