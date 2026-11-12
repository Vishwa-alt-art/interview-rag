import os
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

from fastapi import FastAPI

app = FastAPI()




model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#load the env variable
BASE_URL = Path(__file__).resolve().parent
load_dotenv(BASE_URL / '.env')
api_key = BASE_URL / 'GROQ_API_KEY'

groq_client = Groq(api_key=api_key)

client = chromadb.PersistentClient(path=str(BASE_URL/'chromadb'))
collection = client.get_or_create_document(name="resume_collector")

def extracttext_pdf(pdf_file):

    page_texts = []
    

    reader = PdfReader(pdf_file)
    for page_num, page in enumerate( reader.pages):
        page_text = page.extract_text()
        if page_text.strip():
            page_texts.append({'text': page_text, 'page_number':page_num+1})

    #text splitter
    text_splitter = RecursiveCharacterTextSplitter(
         chunk_size = 100,
         chunk_overlap  = 0,
    
    )

    chunks = []
    metadata = []

    for page in page_texts:
        page_chunks = text_splitter.split_text(page['text'])
        for p in page_chunks:
            chunks.append(p)
            metadata.append({'source_file':Path.pdf_file.name, 'page_number':page_num})
    return chunks, metadata

#convert the text into vectors:
def chunks_into_vectors(pdf_file):
    chunks, metadata =  extracttext_pdf(pdf_file)
    embedding_model = model.encode(chunks)

    
@app.post('/ingest')
def ingest():
    pdf_file = BASE_URL / 'Resume_Vishwa.pdf'
    result = extracttext_pdf(pdf_file)
    return result

def retrieve_context(question):
    query_embedding = model.encode(question)

    #retrieve 3 similar chunks
    k = min(3, collection.count())
    result = collection.query(query_embedding=query_embedding.tolist(),n_results=k, include=['embeddings','documents','metadatas'])

    retrieved_chunks = result['documents'][0]
    retrieved_embeddings = result['embeddings'][0]
    retrieved_metadatas = result['metadatas'][0]

    context = '\n'.join(retrieved_chunks)

    return context, retrieved_embeddings, retrieved_metadatas

def generate_answer(question, context):

    completion = groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
      {
        "role": "system",
        "content": "You are an assistant who gives answer to the user's question and return the answer from the user's resume."
      },
      {
          'role':'user',
          'content': f'''Context: {context}, Question: {question} '''
      }

    ],
    temperature=0.2,
    
    )

    answer = completion.choices[0].message.content
    return answer









    


    
    