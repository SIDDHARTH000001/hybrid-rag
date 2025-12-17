import streamlit as st
import requests
import json
from pathlib import Path
import time
import asyncio
import websockets

from backend.config_manager import get_config

config = get_config()
host = config.server.host
port = config.server.port

BACKEND_URL = f"http://{host}:{port}"
BACKEND_WS = f"ws://{host}:{port}"

st.set_page_config(
    page_title="RAG Pipeline - Document Q&A",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .reference-box {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .status-box {
        background-color: #e7f3ff;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        border-left: 4px solid #2196F3;
        margin: 0.5rem 0;
        font-style: italic;
        color: #0066cc;
    }
    .status-thinking {
        background-color: #fff4e6;
        border-left: 4px solid #ff9800;
        color: #e65100;
    }
    .status-searching {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
    }
    .status-connected {
        color: #28a745;
        font-weight: bold;
    }
    .status-disconnected {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def upload_document(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=100)
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get("detail", "Unknown error")
            raise Exception(error_detail)
    except Exception as e:
        raise Exception(f"Upload failed: {str(e)}")


def clear_document():
    try:
        response = requests.delete(f"{BACKEND_URL}/document", timeout=10)
        return response.status_code == 200
    except:
        return False


async def query_with_websocket(question: str, status_placeholder, answer_placeholder, references_container):
    uri = f"{BACKEND_WS}/ws/query"
    
    full_answer = ""
    references = []
    
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({"question": question}))
            
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                msg_data = data.get("data")
                
                if msg_type == "status":
                    if "Thinking" in msg_data:
                        status_placeholder.markdown(
                            f'<div class="status-box status-thinking">{msg_data}</div>',
                            unsafe_allow_html=True
                        )
                    elif "knowledge base" in msg_data:
                        status_placeholder.markdown(
                            f'<div class="status-box status-searching">{msg_data}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        status_placeholder.info(msg_data)
                
                elif msg_type == "context":
                    status_placeholder.success(f"📄 {msg_data.get('message', 'Found relevant content')}")
                
                elif msg_type == "answer_chunk":
                    full_answer += msg_data
                    answer_placeholder.markdown(full_answer + "▌")
                
                elif msg_type == "references":
                    references = msg_data
                    answer_placeholder.markdown(full_answer)  # Remove cursor
                    
                    if references:
                        with references_container:
                            st.markdown("---")
                            st.markdown("### 📎 Context")
                            for ref in references:
                                st.markdown(f'<div class="reference-box">{ref}</div>', unsafe_allow_html=True)
                
                elif msg_type == "complete":
                    status_placeholder.empty()
                    break
                
                elif msg_type == "error":
                    status_placeholder.error(f"❌ Error: {msg_data}")
                    break
    
    except Exception as e:
        status_placeholder.error(f"❌ Connection error: {str(e)}")
    
    return full_answer, references


def main():
    st.markdown('<div class="main-header"> RAG Pipeline - Document Q&A</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Hybrid Search (BM25 + Semantic) powered by Gemini</div>', unsafe_allow_html=True)
    
    backend_healthy = check_backend_health()
    
    with st.sidebar:
        st.header("⚙️ Backend Connection")
        
        if backend_healthy:
            st.markdown('<p class="status-connected">● Connected</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-disconnected">● Disconnected</p>', unsafe_allow_html=True)
            st.error("Backend is not running. Please start the backend server:")
            st.code("python main.py", language="bash")
            st.stop()
        
        backend_status = get_backend_status()
        
        st.markdown("---")
        
        st.header("📁 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=["pdf", "docx"],
            help="Upload a document to enable Q&A"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        result = upload_document(uploaded_file)
                        st.success(f"{result['message']}")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"{str(e)}")
        
        st.markdown("---")
        st.header("📄 Document Status")
        
        if backend_status and backend_status['document_loaded']:
            st.success(f"Loaded: {backend_status['current_document']}")
            st.info(f"Total chunks: {backend_status['total_chunks']}")
            
            if st.button("🗑️ Clear Document"):
                if clear_document():
                    st.success("Document cleared")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to clear document")
        else:
            st.warning(" No document loaded")
        
        
        st.markdown("---")
        st.header("Info")
        st.markdown("""
        **Features:**
        - Hybrid Search (BM25 + Semantic)
        - Intelligent tool use
        - Detailed references
        - Real-time status updates
        - WebSocket streaming
        """)
    
    if not backend_status or not backend_status['document_loaded']:
        st.info(" Please upload a document from the sidebar to get started")
        
        st.markdown("### Once you upload a document, you can:")
        st.markdown("""
        - **Ask about the document**:
            - "tell me about EON Velvet?"
            - "Which single-door refrigerator series offers Turbo Cooling Technology, and what is its maximum farm-freshness duration?"
            - "Which refrigerator series provides farm-fresh fruits and vegetables for up to 30 days, and what technologies enable this?"
        
        The AI will automatically decide whether to search the document or answer directly!
        
        **Real-time Status:**
        - **Thinking...** - AI is processing your question
        - **Looking into knowledge base...** - AI is searching the document
        - **Found X sections** - Relevant content retrieved
        """)
        
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if "references" in message and message["references"]:
                    st.markdown("---")
                    st.markdown("### References")
                    for ref in message["references"]:
                        st.markdown(f'<div class="reference-box">{ref}</div>', unsafe_allow_html=True)
        
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                answer_placeholder = st.empty()
                references_container = st.container()
                
                answer, references = asyncio.run(
                    query_with_websocket(prompt, status_placeholder, answer_placeholder, references_container)
                )
                
                if answer:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "references": references
                    })


if __name__ == "__main__":
    main()
