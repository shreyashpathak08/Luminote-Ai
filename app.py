import os
import json
import uuid
import re
import warnings
import logging
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from youtube_transcript_api import YouTubeTranscriptApi

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

raw_key = "ADD API HERE" 
os.environ["GOOGLE_API_KEY"] = raw_key.strip()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

HISTORY_FILE = "chat_history.json"

print("Booting up HuggingFace Model into memory...")
global_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# THE UPGRADE: Persistent Multi-Document Brain
DB = None
if os.path.exists("./chroma_db"):
    print("Loading existing Multi-Document Brain...")
    DB = Chroma(persist_directory="./chroma_db", embedding_function=global_embedding)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def extract_youtube_id(url):
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def ingest_text_chunks(text_content):
    global DB
    if not text_content.strip():
        return False
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text_content)
    
    # THE UPGRADE: Append to existing brain instead of overwriting!
    if DB is None:
        DB = Chroma.from_texts(texts=chunks, embedding=global_embedding, persist_directory="./chroma_db")
    else:
        DB.add_texts(texts=chunks)
    return True

def initialize_database(pdf_path):
    if not os.path.exists(pdf_path):
        return False
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception:
        return False
    return ingest_text_chunks(full_text)

def ingest_youtube_url(video_url):
    video_id = extract_youtube_id(video_url)
    if not video_id:
        return False, "Invalid YouTube URL format."
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript_list])
        success = ingest_text_chunks(transcript_text)
        return success, "Lecture transcript successfully indexed!"
    except Exception as e:
        return False, f"Could not extract transcript: {str(e)}"

def query_rag(query, persona_choice, custom_instructions):
    context = "No specific textbook context provided. Rely entirely on your elite general academic knowledge."
    
    if DB is not None:
        try:
            results = DB.similarity_search(query, k=10)
            if results:
                context = "\n\n".join([doc.page_content for doc in results])
        except Exception:
            pass
    
    if persona_choice == "1":
        style = "Explain Like I'm 5: Use simple metaphors and no technical jargon. Break it down easily."
    elif persona_choice == "2":
        style = "Exam Crunch: Provide strict definitions, high-yield bullet points, and key points to memorize."
    else:
        style = "Standard Learner: Provide clear, balanced, and professional educational explanations."

    template = """
    You are Luminote, an elite, highly intelligent academic AI study tool. 
    Style instruction: {style}
    
    CRITICAL CUSTOM USER INSTRUCTIONS: 
    {custom_instructions}

    Context:
    {context}

    Student's Input: {question}

    STRICT INSTRUCTIONS:
    1. GREETING OVERRIDE: If the student's input is just a casual greeting (like "Hello", "Hi", "Hey"), completely ignore the textbook context. Warmly introduce yourself as Luminote, the elite AI study tutor, briefly explain that you are here to dissect notes and crush exams, and ask what they want to study today.
    2. ACADEMIC QUESTIONS: If the input is an actual academic question, enforce the ZERO FLUFF rule. NEVER say "Hello", "Based on the text", or "According to the document". Jump directly into the factual answer in the very first sentence. Treat the context as absolute universal fact.
    3. BE EXHAUSTIVE: Extract every single detail, definition, and concept related to the question.
    4. Use professional Markdown (bolding, headers, bullet points) to structure the data cleanly.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "style": style, 
        "custom_instructions": custom_instructions if custom_instructions else "None", 
        "context": context, 
        "question": query
    })

@app.route("/")
def index():
    new_id = str(uuid.uuid4())[:8]
    return redirect(url_for('chat_session', session_id=new_id))

@app.route("/c/<session_id>", methods=["GET", "POST"])
def chat_session(session_id):
    history = load_history()
    sidebar_state = request.cookies.get('sidebar_collapsed', 'false')
    
    if session_id not in history:
        history[session_id] = {"title": "New Chat", "messages": [], "documents": []}
        
    upload_status = None
    
    if request.method == "POST":
        if 'pdf_file' in request.files:
            file = request.files['pdf_file']
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                if initialize_database(filepath):
                    history[session_id].setdefault("documents", []).append(file.filename)
                    history[session_id]["latest_pdf"] = file.filename
                    doc_count = len(history[session_id]["documents"])
                    upload_status = f"✅ '{file.filename}' added! (Total Sources: {doc_count})"

        youtube_url = request.form.get("youtube_url", "").strip()
        if youtube_url:
            success, msg = ingest_youtube_url(youtube_url)
            if success:
                history[session_id].setdefault("documents", []).append("YouTube Lecture")
                doc_count = len(history[session_id]["documents"])
                upload_status = f"✅ YouTube Lecture Indexed! (Total Sources: {doc_count})"
            else:
                upload_status = f"⚠️ {msg}"
                    
        question = request.form.get("question", "").strip()
        persona = request.form.get("persona", "3")
        custom_instructions = request.form.get("custom_instructions", "").strip()
        
        answer = ""
        if question:
            if len(history[session_id]["messages"]) == 0:
                history[session_id]["title"] = question[:25] + "..." if len(question) > 25 else question
                
            try:
                answer = query_rag(question, persona, custom_instructions)
            except Exception as e:
                answer = f"Error: {e}"
                
            history[session_id]["messages"].append({"q": question, "a": answer})
            save_history(history)
            
        if request.headers.get("X-Fetch-API") == "true":
            return jsonify({
                "success": True,
                "q": question,
                "a": answer,
                "upload_status": upload_status,
                "doc_count": len(history[session_id].get("documents", [])),
                "latest_pdf": history[session_id].get("latest_pdf"),
                "title": history[session_id]["title"]
            })
            
    doc_count = len(history[session_id].get("documents", []))
    latest_pdf = history[session_id].get("latest_pdf")
    
    return render_template("index.html", 
                           chat_log=history[session_id]["messages"], 
                           sessions=history, 
                           current_session=session_id,
                           upload_status=upload_status,
                           doc_count=doc_count,
                           latest_pdf=latest_pdf,
                           sidebar_state=sidebar_state)

@app.route("/generate_flashcards/<session_id>", methods=["POST"])
def generate_flashcards(session_id):
    if DB is None:
        return jsonify({"success": False, "error": "No study materials or documents loaded yet. Please upload notes or a YouTube lecture first!"})
    
    try:
        sample_chunks = DB.similarity_search("core definitions and key summary concepts", k=8)
        context = "\n\n".join([c.page_content for c in sample_chunks])
        
        prompt = ChatPromptTemplate.from_template("""
        You are an exam preparation expert. Based on the following study materials:
        {context}

        Generate exactly 8 high-yield flashcards focusing on testable definitions, core formulas, and crucial distinctions.
        Format your response strictly as valid raw JSON array of objects with "question" and "answer" keys.
        Do NOT wrap in markdown backticks. Example:
        [
          {{"question": "What is concept X?", "answer": "Concept X is defined as..."}},
          {{"question": "How does A differ from B?", "answer": "A is... while B is..."}}
        ]
        """)
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
        chain = prompt | llm | StrOutputParser()
        raw_res = chain.invoke({"context": context}).strip()
        
        if raw_res.startswith("```"):
            raw_res = re.sub(r"^```[a-zA-Z]*\n", "", raw_res)
            raw_res = re.sub(r"\n```$", "", raw_res)
            
        flashcards = json.loads(raw_res)
        return jsonify({"success": True, "flashcards": flashcards})
    except Exception as e:
        return jsonify({"success": False, "error": f"Flashcard generation failed: {str(e)}"})

@app.route("/rename/<session_id>", methods=["POST"])
def rename_chat(session_id):
    history = load_history()
    new_title = request.form.get("new_title", "New Chat").strip()
    if session_id in history and new_title:
        history[session_id]["title"] = new_title[:40] 
        save_history(history)
    return redirect(url_for('chat_session', session_id=session_id))

@app.route("/delete/<session_id>", methods=["POST"])
def delete_chat(session_id):
    history = load_history()
    if session_id in history:
        del history[session_id]
        save_history(history)
    return redirect(url_for('index'))

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)