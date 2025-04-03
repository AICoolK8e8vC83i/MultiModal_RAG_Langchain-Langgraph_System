# 🔮 Multimodal RAG + LangGraph Cognitive Pipeline (State-Aware + Memory + Fallbacks using GPT-4 Vision + Audio)

from langchain_core.tools import Tool 
from langgraph.graph import StateGraph, END 
from langchain.chains import RetrievalQA 
from langchain_community.vectorstores import FAISS 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import base64
import requests
from PIL import Image
import tempfile

# === Load .env ===
load_dotenv()

# === Embedding Models ===
text_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === GPT-4 Vision Image Captioning ===
from openai import OpenAI 

client = OpenAI() 

def describe_image(state):
    image_path = state["input"]

    # read and encode the image
    with open(image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode("utf-8")

    # Use GPT-4 Vision model
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image? Describe it with high detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ],
            }
        ],
        max_tokens=500,
    )

    caption = response.choices[0].message.content
    memory["entries"].append(f"🖼️ Image Caption: {caption}")
    return {"input": caption, "memory_log": memory["entries"], "file_type": "image"}

# === GPT Whisper Audio to Text ===
def transcribe_audio(state):
    file_path = state["input"]
    with open(file_path, "rb") as audio_file:
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    transcription = transcript_response
    memory["entries"].append(f"🗣️ Transcription: {transcription}")
    return {"input": transcription, "memory_log": memory["entries"], "file_type": "audio"}

# === Multilingual Translation (via GPT) ===
def translate_to_english(state):
    text = state["input"]
    llm = ChatOpenAI(model_name="gpt-4-1106-preview")
    prompt = f"Translate the following text to English: {text}"
    translated = llm.invoke(prompt)
    memory["entries"].append(f"🌐 Translation: {translated}")
    return {"input": translated, "memory_log": memory["entries"], "file_type": "translation"}

# === Markdown Loader ===
def load_markdown_files(base_paths):
    documents = []
    for base_path in base_paths:
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(".md"):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        documents.append({"text": content, "metadata": {"source": full_path}})
    return documents

# === Memory State ===
memory = {"entries": []}

# === FAISS RAG Setup ===
base_paths = ['Insert Folders Containing .md Files']

docs_raw = load_markdown_files(base_paths)
documents = [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in docs_raw]
text_vectorstore = FAISS.from_documents(documents, embedding=text_embedder)
text_vectorstore.save_local("text_index")
text_retriever = text_vectorstore.as_retriever()

llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.9)

text_rag = RetrievalQA.from_chain_type(llm=llm, retriever=text_retriever)

def rag_with_memory(state):
    question = state["input"]
    response = text_rag.run(question)
    memory["entries"].append(f"🧠 RAG Answer: {response}")
    return {"input": response, "memory_log": memory["entries"], "file_type": "text"}

# === LangGraph Routing ===
from typing import TypedDict, Union

class AgentState(TypedDict):
    input: Union[str, dict]
    memory_log: list[str]
    file_type: str

def multimodal_router(state: AgentState):
    content = state["input"]
    if isinstance(content, str) and content.endswith(".wav") or content.endswith(".mp3"):
        next_node = "transcribe"
    elif isinstance(content, str) and content.endswith(".jpg") or content.endswith(".png"):
        next_node = "describe"
    elif isinstance(content, str):
        next_node = "qa"
    else:
        next_node = END
    return {"input": content, "memory_log": memory["entries"], "file_type": "router", "next": next_node}

workflow = StateGraph(AgentState)
workflow.add_node("transcribe", transcribe_audio)
workflow.add_node("describe", describe_image)
workflow.add_node("translate", translate_to_english)
workflow.add_node("qa", rag_with_memory)
workflow.add_node("router", multimodal_router)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: x["next"])
graph = workflow.compile()
