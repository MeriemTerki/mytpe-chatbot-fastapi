# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langdetect import detect
from googletrans import Translator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import huggingface_hub

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    chat_history: list = []

# Load the chatbot model and set up the pipeline
device = 'cuda' if torch.cuda.is_available() else 'cpu'

origin_model_path = "mistralai/Mistral-7B-Instruct-v0.1"
model_path = "filipealmeida/Mistral-7B-Instruct-v0.1-sharded"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                            quantization_config=bnb_config,
                                            device_map="auto",
                                            token='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')
tokenizer = AutoTokenizer.from_pretrained(origin_model_path, token='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')

text_generation_pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=300,
    temperature=0.3,
    do_sample=True,
)
mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Load CSV data for chatbot
loader = CSVLoader(file_path='mytpe-dataset.csv')
data = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunked_docs = text_splitter.split_documents(data)

# Create embeddings and database for retrieval
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(chunked_docs,
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

# Connect query to FAISS index using a retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}
)

# Create the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(mistral_llm, retriever, return_source_documents=True)


@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.query
    chat_history = request.chat_history

    if query.strip() == '':
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query = "Veuillez répondre en français: " + query
    result = qa_chain.invoke({'question': query, 'chat_history': chat_history})

    response = result['answer']
    response_language = detect(response)

    if response_language != 'fr':
        translator = Translator()
        response = translator.translate(response)

    chat_history.append((query, response))
    return {"answer": response, "chat_history": chat_history}
