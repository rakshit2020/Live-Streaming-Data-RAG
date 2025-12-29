# Live-Streaming-Data-RAG

## I Built a RAG System That Listens to Live BBC News and Answers Questions About What Happened 10 Minutes Ago (read more - https://huggingface.co/blog/RakshitAralimatti/streaming-data-rag)


## The Problem Nobody Talks About in RAG

Every RAG tutorial shows you how to query static documents. Upload PDFs, chunk them, embed them, done. But what if your knowledge base is constantly changing? What if information flows in real-time and you need to ask "what happened in the last 30 minutes?"

Traditional RAG breaks down completely. You cannot ask temporal questions like "what was the breaking news at 9 AM?" or "summarize channel 0 from the past hour" because documents have no concept of time.

I spent a weekend fixing this.

## What I Built

A live audio streaming RAG system that continuously captures BBC World Service radio (http://stream.live.vc.bbcmedia.co.uk/bbc_world_service), transcribes it in real-time, and lets you query across temporal windows with natural language.

Not just semantic search. Time-aware semantic search.

Ask it "what were the main topics in the last 10 minutes" and it filters documents by timestamp, retrieves relevant chunks, reranks them for accuracy, and generates an answer citing specific broadcast times. Every answer includes sources with precise UTC timestamps like "In the 14:23 UTC segment..."

The system runs 24/7 in the background, capturing 60-second audio chunks, transcribing them with NVIDIA Riva ASR, embedding them with NeMo Retriever, and indexing them with temporal metadata. Within seconds of broadcast, the content becomes queryable.

## How It Actually Works

Think of it as three parallel processes that never stop:

**The Listening Loop:** Every minute, FFmpeg captures live audio from BBC World Service. The chunk gets saved with a UTC timestamp in its filename. No audio is lost. No gaps.

**The Intelligence Layer:** NVIDIA Riva transcribes each audio chunk into text. The transcript gets embedded using NeMo Retriever's 300M parameter model and stored in ChromaDB. But here is the key: every document carries metadata with Unix timestamps for when the audio started and ended.

**The Query Engine:** When you ask a question, the system first applies time filters. "Last 30 minutes" translates to a database filter on Unix timestamps. Then vector search happens only within that time window. NVIDIA's Llama 3.2 reranker scores the top candidates. Ministral 14B generates the final answer using only those time-filtered, reranked sources.

The result is a conversational interface where time is a first-class citizen. Not an afterthought.

## Why This Matters Beyond News

I chose BBC World Service for the demo because it is reliable and publicly accessible. But the architecture was inspired by NVIDIA's Software-Defined Radio blueprint for defense and intelligence applications.

Imagine monitoring emergency radio frequencies during natural disasters. First responders need to ask "what happened on channel 3 between 2 AM and 4 AM?" when coordinating rescue operations.

Think about financial compliance. Trading floors could index all voice communications with temporal audit trails. Regulators could query "show me all conversations mentioning this stock ticker in the last hour."

Intelligence agencies already monitor multiple radio frequencies simultaneously. Now imagine querying those feeds with natural language and getting answers that cite exact broadcast times across multiple channels.

The system supports multi-channel ingestion. You can capture dozens of streams concurrently, each maintaining independent temporal indexes while sharing a unified query interface.

## Installization

```
apt-get update -qq && apt-get install -y ffmpeg
pip install -q streamlit
pip install -q nvidia-riva-client
pip install -q langchain langchain-nvidia-ai-endpoints langchain-chroma chromadb
git clone --depth 1 https://github.com/nvidia-riva/python-clients.git /content/python-clients 2>/dev/null || echo "Already cloned"

print("✅ Installation complete!")
```
## The Tech Stack
Built entirely on NVIDIA NIM microservices for production-grade performance. Riva handles automatic speech recognition, NeMo Retriever 300M generates embeddings, Llama 3.2 1B reranks for accuracy, and Ministral 14B generates final answers. ChromaDB stores vectors with temporal metadata. LangChain orchestrates the pipeline. Streamlit provides the interface.

Background capture runs continuously in a separate thread, indexing new content while you query existing data. The system never stops learning.
