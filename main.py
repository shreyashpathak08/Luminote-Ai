import os
import json
import uuid
import re
import warnings
import logging
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from youtube_transcript_api import YouTubeTranscriptApi

# TERMINAL SILENCERS
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# GROQ API CONFIGURATION
raw_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
os.environ["GROQ_API_KEY"] = raw_key.strip()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMPLATES_AUTO_RELOAD'] = True
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

HISTORY_FILE = "chat_history.json"

print("Booting up HuggingFace Model into memory...")
global_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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

def ingest_text_chunks(text_content, session_id, source_name="Document"):
    global DB
    if not text_content.strip():
        return False
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_text(text_content)
    
    metadatas = [{"session_id": session_id, "source": source_name} for _ in chunks]
    
    if DB is None:
        DB = Chroma.from_texts(texts=chunks, embedding=global_embedding, persist_directory="./chroma_db", metadatas=metadatas)
    else:
        DB.add_texts(texts=chunks, metadatas=metadatas)
    return True

def initialize_database(pdf_path, session_id):
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
    filename = os.path.basename(pdf_path)
    return ingest_text_chunks(full_text, session_id, source_name=filename)

def ingest_youtube_url(video_url, session_id):
    video_id = extract_youtube_id(video_url)
    if not video_id:
        return False, "Invalid YouTube URL format."
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript_list])
        success = ingest_text_chunks(transcript_text, session_id, source_name=f"YouTube ({video_id})")
        return success, "Lecture transcript successfully indexed!"
    except Exception as e:
        return False, f"Could not extract transcript: {str(e)}"

def query_rag(query, persona_choice, custom_instructions, session_id, target_source=None):
    context = "No specific study material provided. Rely entirely on your elite general academic knowledge."
    sources_used = []
    
    if DB is not None:
        try:
            search_filter = {"session_id": session_id}
            
            # STRICT ANTI-MIXING FILTER
            if target_source and target_source != "ALL":
                search_filter = {
                    "$and": [
                        {"session_id": session_id},
                        {"source": target_source}
                    ]
                }
                
            # k=3 fixes the 503 API overload error while maintaining accuracy
            results = DB.similarity_search(query, k=3, filter=search_filter)

            if results:
                context_chunks = []
                for doc in results:
                    src = doc.metadata.get('source', 'Material')
                    if src not in sources_used:
                        sources_used.append(src)
                    context_chunks.append(f"[{src}]: {doc.page_content}")
                context = "\n\n".join(context_chunks)
        except Exception as e:
            print(f"Chroma Search Error: {e}")
            pass
    
    if persona_choice == "1":
        style = "Explain Like I'm 5: Use simple analogies, plain English, and zero complex jargon."
    elif persona_choice == "2":
        style = "Exam Crunch: High-yield revision format with strict definitions, critical formulas, and high-impact takeaways."
    elif persona_choice == "4":
        style = "Strict Examiner: You are an exam simulator. Ask ONE challenging question based strictly on the content. Wait for their answer."
    else:
        style = "Standard Learner: Comprehensive, polished, and structured academic explanations."

    template = """
    You are Luminote, an elite academic AI study engine. 
    Pedagogical Style: {style}
    
    CRITICAL USER INSTRUCTIONS: 
    {custom_instructions}

    Study Context:
    {context}

    Student's Input: {question}

    EXECUTION PROTOCOL:
    1. SOURCE ACCURACY: Rely strictly on the study context provided. If asked about a specific chapter, answer using the core lesson content of that chapter.
    2. ANTI-FRONTMATTER RULE: Ignore book prefaces, syllabus matrices, and publication notes. Focus exclusively on actual topics.
    3. ZERO FLUFF: Jump straight into the factual content on sentence one. Do not use filler openers.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_retries=1, timeout=20)
    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "style": style, 
        "custom_instructions": custom_instructions if custom_instructions else "None", 
        "context": context, 
        "question": query
    })
    
    return answer, len(sources_used)

@app.route("/")
def index():
    new_id = str(uuid.uuid4())[:8]
    return redirect(url_for('chat_session', session_id=new_id))

@app.route("/c/<session_id>", methods=["GET", "POST"])
def chat_session(session_id):
    history = load_history()
    
    if session_id not in history:
        history[session_id] = {"title": "New Chat", "messages": [], "documents": []}
        
    upload_status_list = []
    
    if request.method == "POST":
        if 'pdf_file' in request.files:
            file = request.files['pdf_file']
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                if initialize_database(filepath, session_id):
                    if file.filename not in history[session_id].get("documents", []):
                        history[session_id].setdefault("documents", []).append(file.filename)
                    history[session_id]["latest_pdf"] = file.filename
                    save_history(history)
                    upload_status_list.append(f"'{file.filename}'") 

        youtube_url = request.form.get("youtube_url", "").strip()
        if youtube_url:
            success, msg = ingest_youtube_url(youtube_url, session_id)
            if success:
                vid_id = extract_youtube_id(youtube_url)
                yt_name = f"YouTube ({vid_id})"
                if yt_name not in history[session_id].get("documents", []):
                    history[session_id].setdefault("documents", []).append(yt_name)
                upload_status_list.append(f"YouTube lecture ({vid_id})")

        question = request.form.get("question", "").strip()
        persona = request.form.get("persona", "3")
        custom_instructions = request.form.get("custom_instructions", "").strip()
        target_source = request.form.get("target_source", "ALL")
        
        answer = ""
        source_count = 0
        if question:
            if len(history[session_id]["messages"]) == 0:
                history[session_id]["title"] = question[:28] + "..." if len(question) > 28 else question
                
            is_error = False
            try:
                answer, source_count = query_rag(question, persona, custom_instructions, session_id, target_source)
            except Exception as e:
                answer = f"Error: {e}"
                is_error = True
                
            if not is_error:
                history[session_id]["messages"].append({
                    "q": question, 
                    "a": answer,
                    "sources": source_count
                })
                save_history(history)
            
        doc_count = len(history[session_id].get("documents", []))
        upload_status = f"✅ Ingested {', '.join(upload_status_list)}!" if upload_status_list else None
            
        if request.headers.get("X-Fetch-API") == "true":
            return jsonify({
                "success": True,
                "q": question,
                "a": answer,
                "sources": source_count,
                "upload_status": upload_status,
                "doc_count": doc_count,
                "documents": history[session_id].get("documents", []),
                "latest_pdf": history[session_id].get("latest_pdf"),
                "title": history[session_id]["title"]
            })
            
    doc_count = len(history[session_id].get("documents", []))
    latest_pdf = history[session_id].get("latest_pdf")
    
    return render_template("index.html", 
                           chat_log=history[session_id]["messages"], 
                           sessions=history, 
                           current_session=session_id,
                           upload_status=None,
                           doc_count=doc_count,
                           documents=history[session_id].get("documents", []),
                           latest_pdf=latest_pdf)

@app.route("/generate_flashcards/<session_id>", methods=["POST"])
def generate_flashcards(session_id):
    if DB is None:
        return jsonify({"success": False, "error": "No study materials loaded yet."})
    
    try:
        sample_chunks = DB.similarity_search("key concepts definitions formulas summary", k=4, filter={"session_id": session_id})
        if not sample_chunks:
            return jsonify({"success": False, "error": "No materials indexed for this chat yet."})
            
        context = "\n\n".join([c.page_content for c in sample_chunks])
        
        prompt = ChatPromptTemplate.from_template("""
        You are an elite academic tutor. Based on this material:
        {context}

        Produce exactly 6 high-yield revision flashcards.
        Output MUST be pure JSON with NO markdown code-fences, exactly in this structure:
        [
          {{"question": "Front question here", "answer": "Back definition or answer here"}}
        ]
        """)
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_retries=1, timeout=20)
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
    app.run(host='0.0.0.0', port=5000, debug=True)