import streamlit as st
import subprocess
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import time
import threading
from collections import deque
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_chroma import Chroma
from langchain_core.documents import Document
from  langchain_classic.retrievers import ContextualCompressionRetriever

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Live News Intelligence", page_icon="📡", layout="wide")

NVIDIA_API_KEY = "Enter Your KEY"
FUNCTION_ID = "71203149-d3b7-4460-8231-1be2543a1fca"
GRPC_SERVER = "grpc.nvcf.nvidia.com:443"
BBC_STREAM_URL = "http://stream.live.vc.bbcmedia.co.uk/bbc_world_service"
CHUNK_DURATION = 60
CHANNEL_ID = 0

BASE_DIR = Path("/content/streaming_rag")
AUDIO_DIR = BASE_DIR / "audio_chunks"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
METADATA_DIR = BASE_DIR / "metadata"
CHROMA_DIR = BASE_DIR / "chroma_langchain"

for d in [AUDIO_DIR, TRANSCRIPT_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
knowledge_base = deque(maxlen=1000)

# ============================================================================
# INITIALIZE COMPONENTS (Added Reranker!)
# ============================================================================
@st.cache_resource
def init_components():
    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nemoretriever-300m-embed-v2", 
        truncate="END"
    )
    
    llm = ChatNVIDIA(
        model="mistralai/ministral-14b-instruct-2512", 
        temperature=0.2, 
        max_completion_tokens=200
    )
    
    vectorstore = Chroma(
        collection_name="bbc_streaming_rag", 
        embedding_function=embeddings, 
        persist_directory=str(CHROMA_DIR)
    )
    
    # NEW: Initialize NVIDIA Reranker
    reranker = NVIDIARerank(
        model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        top_n=5  # Keep top 5 after reranking
    )
    
    return embeddings, llm, vectorstore, reranker

embeddings, llm, vectorstore, reranker = init_components()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def index_transcript_langchain(metadata: dict) -> str:
    chunk_id = metadata["chunk_id"]
    transcript = metadata["transcript"]
    doc = Document(
        page_content=transcript,
        metadata={k: v for k, v in metadata.items() if k not in ['transcript']}
    )
    vectorstore.add_documents([doc], ids=[chunk_id])
    return chunk_id

def build_time_filter(minutes_ago: int = None, time_window_minutes: int = 5, channel_id: int = None):
    conditions = []
    if channel_id is not None:
        conditions.append({"channel_id": {"$eq": channel_id}})
    if minutes_ago is not None:
        now_unix = int(datetime.now(timezone.utc).timestamp())
        target_unix = now_unix - (minutes_ago * 60)
        window_start = target_unix - (time_window_minutes * 60 // 2)
        window_end = target_unix + (time_window_minutes * 60 // 2)
        conditions.append({"unix_start": {"$lte": window_end}})
        conditions.append({"unix_end": {"$gte": window_start}})
    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}

# ============================================================================
# NEW: ENHANCED QUERY WITH RERANKING
# ============================================================================
def query_streaming_rag_with_rerank(
    query_text: str, 
    channel_id: int = None, 
    minutes_ago: int = None, 
    time_window_minutes: int = 5, 
    initial_k: int = 20,  # Retrieve more initially
    top_k: int = 5        # Keep top 5 after reranking
):
    """
    Two-stage retrieval: 
    1. Vector search retrieves initial_k candidates
    2. Reranker narrows down to top_k most relevant
    """
    where_filter = build_time_filter(minutes_ago, time_window_minutes, channel_id)
    
    # Stage 1: Initial retrieval (cast to wider net)
    initial_docs = vectorstore.similarity_search(
        query_text, 
        k=initial_k, 
        filter=where_filter
    )
    
    if not initial_docs:
        return []
    
    # Stage 2: Rerank with NVIDIA model
    reranked_docs = reranker.compress_documents(
        documents=initial_docs,
        query=query_text
    )
    
    # Return top_k after reranking
    return reranked_docs[:top_k]

# ============================================================================
# CORE CLASS (Same as before)
# ============================================================================
class StreamingDataRAG:
    def __init__(self):
        self.counter_file = BASE_DIR / "chunk_counter.txt"
        self.chunk_counter = int(self.counter_file.read_text()) if self.counter_file.exists() else 0
        self.session_start = datetime.now(timezone.utc)
    
    def _save_counter(self):
        self.counter_file.write_text(str(self.chunk_counter))
    
    def capture_audio_chunk(self) -> Path:
        self.chunk_counter += 1
        self._save_counter()
        timestamp_utc = datetime.now(timezone.utc)
        ts_str = timestamp_utc.strftime("%Y%m%dT%H%M%SZ")
        chunk_file = AUDIO_DIR / f"chunk_{self.chunk_counter:04d}_{ts_str}.wav"
        cmd = ["ffmpeg", "-y", "-i", BBC_STREAM_URL, "-t", str(CHUNK_DURATION), 
               "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(chunk_file)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=CHUNK_DURATION + 30, check=True)
            return chunk_file
        except:
            return None
    
    def transcribe_with_riva(self, audio_file: Path) -> str:
        cmd = ["python", "/content/python-clients/scripts/asr/transcribe_file.py", 
               "--server", GRPC_SERVER, "--use-ssl",
               "--metadata", "function-id", FUNCTION_ID, 
               "--metadata", "authorization", f"Bearer {NVIDIA_API_KEY}",
               "--input-file", str(audio_file), 
               "--language-code", "en-US", "--automatic-punctuation"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith("Transcript:"):
                        return line.replace("Transcript:", "").strip()
                return result.stdout.strip()
            return None
        except:
            return None
    
    def save_and_index(self, audio_file: Path, transcript: str) -> dict:
        filename_parts = audio_file.stem.split('_')
        ts_str = filename_parts[-1]
        start_dt = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(seconds=CHUNK_DURATION)
        chunk_id = f"chunk_{self.chunk_counter:04d}"
        metadata = {
            "chunk_id": chunk_id, "channel_id": CHANNEL_ID, "source": "BBC_World_Service",
            "start_time_utc": start_dt.isoformat(), "end_time_utc": end_dt.isoformat(),
            "unix_start": int(start_dt.timestamp()), "unix_end": int(end_dt.timestamp()),
            "duration_seconds": CHUNK_DURATION, "transcript": transcript,
            "char_count": len(transcript), "word_count": len(transcript.split()),
            "audio_file": str(audio_file), "language": "en-US", "asr_model": "parakeet-riva-cloud"
        }
        (TRANSCRIPT_DIR / f"{chunk_id}.txt").write_text(transcript, encoding='utf-8')
        (METADATA_DIR / f"{chunk_id}.json").write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        knowledge_base.append(metadata)
        try:
            index_transcript_langchain(metadata)
        except:
            pass
        return metadata

# ============================================================================
# UPDATED ask_streaming_rag WITH RERANKING
# ============================================================================
def ask_streaming_rag(
    question: str, 
    channel_id: int = None, 
    minutes_ago: int = None, 
    time_window_minutes: int = 5, 
    use_reranking: bool = True,
    top_k: int = 3
) -> dict:
    """
    Enhanced RAG with optional reranking for better accuracy
    """
    # Use reranking if enabled
    if use_reranking:
        docs = query_streaming_rag_with_rerank(
            query_text=question,
            channel_id=channel_id,
            minutes_ago=minutes_ago,
            time_window_minutes=time_window_minutes,
            initial_k=20,  # Retrieve 20 candidates
            top_k=top_k    # Rerank to top K
        )
    else:
        # Fallback to direct vector search
        where_filter = build_time_filter(minutes_ago, time_window_minutes, channel_id)
        docs = vectorstore.similarity_search(question, k=top_k, filter=where_filter)
    
    if not docs:
        return {
            "question": question, 
            "answer": "No relevant transcripts found for the specified time window.", 
            "sources": [],
            "reranked": use_reranking
        }
    
    context_parts = []
    sources = []
    for doc in docs:
        meta = doc.metadata
        time_str = meta['start_time_utc'][11:19]
        context_parts.append(f"[{time_str} UTC - {meta['chunk_id']}]\\n{doc.page_content}")
        sources.append({
            "chunk_id": meta['chunk_id'], 
            "time": meta['start_time_utc'], 
            "preview": doc.page_content[:100]
        })
    
    context = "\\n\\n---\\n\\n".join(context_parts)
    
    # Original prompt (unchanged)
    prompt = f"""You are an AI assistant analyzing BBC World Service news transcripts. Answer the user's question using only the information in the transcripts. If the information is not present, explicitly say "I don't know."

TRANSCRIPTS:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
Be accurate, concise, and directly answer the question.
Do not add extra background or explanations unless they are needed to answer.
If you use information from a specific segment, mention its time (for example, "In the 09:12 UTC segment…").
If the transcripts do not contain the answer, respond with: "I don't know based on the available transcripts." """
    
    response = llm.invoke(prompt)
    return {
        "question": question, 
        "answer": response.content, 
        "sources": sources, 
        "num_sources": len(sources),
        "reranked": use_reranking
    }

# ============================================================================
# BACKGROUND CAPTURE (Same as before)
# ============================================================================
if 'background_running' not in st.session_state:
    st.session_state.background_running = False
    st.session_state.background_thread = None
    st.session_state.capture_log = []

def background_capture_loop():
    rag = StreamingDataRAG()
    while st.session_state.background_running:
        try:
            audio = rag.capture_audio_chunk()
            if audio:
                transcript = rag.transcribe_with_riva(audio)
                if transcript:
                    meta = rag.save_and_index(audio, transcript)
                    log_msg = f"✅ {meta['chunk_id']}: {meta['word_count']} words at {meta['start_time_utc'][11:19]} UTC"
                    st.session_state.capture_log.append(log_msg)
                    if len(st.session_state.capture_log) > 10:
                        st.session_state.capture_log.pop(0)
        except Exception as e:
            st.session_state.capture_log.append(f"❌ Error: {str(e)[:50]}")
        time.sleep(5)

def get_db_stats():
    all_docs = vectorstore._collection.get(include=["metadatas"])
    if not all_docs['ids']:
        return None
    unix_times = [(m['unix_start'], m['unix_end']) for m in all_docs['metadatas']]
    return {
        "total_docs": len(all_docs['ids']),
        "total_words": sum(m.get('word_count', 0) for m in all_docs['metadatas']),
        "oldest_time": datetime.fromtimestamp(min(t[0] for t in unix_times), tz=timezone.utc),
        "newest_time": datetime.fromtimestamp(max(t[1] for t in unix_times), tz=timezone.utc),
        "span_minutes": (max(t[1] for t in unix_times) - min(t[0] for t in unix_times)) / 60
    }

# ============================================================================
# STREAMLIT UI (Enhanced with Reranking Toggle)
# ============================================================================
st.markdown("""<style>
.main-header {font-size: 2.5rem; font-weight: 700; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.stButton>button {background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border: none; 
                   padding: 0.75rem; font-weight: 600; border-radius: 8px;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📡 Live News Intelligence</div>', unsafe_allow_html=True)
st.caption("Real-time Streaming Data to RAG • Powered by NVIDIA NIM • Enhanced with Reranking")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 System Status")
    stats = get_db_stats()
    if stats:
        st.metric("📚 Transcripts", stats['total_docs'])
        st.metric("💬 Words", f"{stats['total_words']:,}")
        st.metric("⏱️ Span", f"{stats['span_minutes']:.1f} min")
        age = (datetime.now(timezone.utc) - stats['newest_time']).total_seconds() / 60
        st.metric("🕐 Latest", f"{age:.1f} min ago")
    else:
        st.info("No data yet")
    
    st.markdown("---")
    st.markdown("### 🔄 Background Capture")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start" if not st.session_state.background_running else "⏸️ Running", 
                     disabled=st.session_state.background_running, type="primary", use_container_width=True):
            st.session_state.background_running = True
            st.session_state.background_thread = threading.Thread(target=background_capture_loop, daemon=True)
            st.session_state.background_thread.start()
            st.rerun()
    with col2:
        if st.button("⏹️ Stop", disabled=not st.session_state.background_running, use_container_width=True):
            st.session_state.background_running = False
            st.rerun()
    
    if st.session_state.background_running:
        st.success("🟢 Live capture active")
    
    if st.session_state.capture_log:
        with st.expander("📜 Capture Log", expanded=True):
            for log in reversed(st.session_state.capture_log[-5:]):
                st.text(log)
    
    st.markdown("---")
    st.markdown("### 🧠 AI Models")
    st.write("**Embeddings:** NeMo 300M")
    st.write("**Reranker:** Llama 3.2 1B")
    st.write("**LLM:** Ministral 14B")

tab1, tab2, tab3 = st.tabs(["🔍 Query", "📡 Manual Capture", "📊 Database"])

with tab1:
    st.markdown("### Ask Questions")
    
    # NEW: Reranking toggle
    use_rerank = st.toggle("🎯 Enable Reranking (Improves Accuracy)", value=True, 
                           help="Uses NVIDIA Llama 3.2 reranker to improve result relevance")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Question", placeholder="What were the main topics?", label_visibility="collapsed")
    with col2:
        time_filter = st.selectbox("Time", ["All Time", "Last 60m", "Last 30m", "Last 10m"])
    
    time_map = {"All Time": None, "Last 60m": 60, "Last 30m": 30, "Last 10m": 10}
    
    if st.button("🔍 Search", type="primary"):
        if question:
            with st.spinner("Analyzing with reranking..." if use_rerank else "Analyzing..."):
                result = ask_streaming_rag(
                    question, 
                    channel_id=0, 
                    minutes_ago=time_map[time_filter], 
                    time_window_minutes=30, 
                    use_reranking=use_rerank,
                    top_k=5
                )
                
                st.markdown("### 💡 Answer")
                if result.get('reranked'):
                    st.info("✨ Results enhanced with NVIDIA Reranker")
                st.success(result['answer'])
                
                if result['sources']:
                    st.markdown("### 📚 Sources")
                    for i, src in enumerate(result['sources'], 1):
                        with st.expander(f"Source {i} - {src['time'][11:19]} UTC - {src['chunk_id']}"):
                            st.write(src['preview'])

with tab2:
    st.markdown("### 🎙️ Manual Capture")
    col1, col2 = st.columns([1, 1])
    with col1:
        n_chunks = st.number_input("Chunks", 1, 5, 1)
    with col2:
        st.write("")
        start = st.button("▶️ Capture Now", type="primary", use_container_width=True)
    
    if start:
        rag = StreamingDataRAG()
        progress = st.progress(0)
        status = st.empty()
        for i in range(n_chunks):
            status.info(f"📡 Capturing {i+1}/{n_chunks}...")
            progress.progress(i / n_chunks)
            audio = rag.capture_audio_chunk()
            if audio:
                status.info(f"📝 Transcribing {i+1}...")
                transcript = rag.transcribe_with_riva(audio)
                if transcript:
                    meta = rag.save_and_index(audio, transcript)
                    with st.expander(f"✅ {meta['chunk_id']} ({meta['word_count']} words)", expanded=True):
                        st.write(transcript[:300])
        progress.progress(1.0)
        status.success(f"✅ Done!")
        time.sleep(2)
        st.rerun()

with tab3:
    st.markdown("### 📊 Database")
    stats = get_db_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📚 Docs", stats['total_docs'])
        col2.metric("💬 Words", f"{stats['total_words']:,}")
        col3.metric("⏱️ Span", f"{stats['span_minutes']:.0f}m")
        col4.metric("📝 Avg", stats['total_words'] // stats['total_docs'])
        st.markdown("---")
        all_docs = vectorstore._collection.get(include=["metadatas", "documents"])
        sorted_docs = sorted(zip(all_docs['ids'], all_docs['metadatas'], all_docs['documents']),
                           key=lambda x: x[1]['unix_start'], reverse=True)[:10]
        for doc_id, meta, text in sorted_docs:
            with st.expander(f"{meta['chunk_id']} - {meta['start_time_utc'][11:19]} UTC ({meta['word_count']} words)"):
                st.write(text)
    else:
        st.info("Database empty")
