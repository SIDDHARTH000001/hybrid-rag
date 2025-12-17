from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import asyncio
from pathlib import Path
import shutil
import json

from backend.config_manager import get_config
from backend.document_processor import DocumentProcessor
from backend.embeddings import get_embedding_model
from backend.vector_store import FAISSVectorStore
from backend.bm25_retriever import BM25Retriever
from backend.hybrid_search import HybridSearcher
from backend.rag_graph import RAGGraph
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting RAG Pipeline Backend...")
    
    state.config = get_config()
    print("Configuration loaded")
    
    print(f"Loading embedding model: {state.config.embeddings.model_name}")
    state.embedding_model = get_embedding_model(
        model_name=state.config.embeddings.model_name,
        device=state.config.embeddings.device
    )

    _ = state.embedding_model.dimension
    print(f"Embedding model loaded ... ")
    
    state.vector_store = FAISSVectorStore(
        dimension=state.embedding_model.dimension,
        index_path=state.config.vector_store.faiss_index_path,
        metadata_path=state.config.vector_store.metadata_path
    )
    print(f" Vector store initialized")
    
    state.bm25_retriever = BM25Retriever()
    print(f"BM25 retriever initialized")
    
    state.hybrid_searcher = HybridSearcher(
        vector_store=state.vector_store,
        bm25_retriever=state.bm25_retriever,
        embedding_model=state.embedding_model,
        bm25_weight=state.config.hybrid_search.bm25_weight,
        semantic_weight=state.config.hybrid_search.semantic_weight
    )
    print(f"Hybrid searcher initialized")
    
    state.rag_graph = RAGGraph(
        hybrid_searcher=state.hybrid_searcher,
        model_name=state.config.llm.model_name,
        provider=state.config.llm.provider,
        endPoint=state.config.llm.endpoint,
        version=state.config.llm.version,
        api_key=state.config.llm.api_key,
        temperature=state.config.llm.temperature,
        max_tokens=state.config.llm.max_tokens,
        top_k=state.config.hybrid_search.top_k
    )
    print(f"RAG graph initialized")
    
    state.document_processor = DocumentProcessor(
        chunk_size=state.config.document_processing.chunk_size,
        chunk_overlap=state.config.document_processing.chunk_overlap,
        min_chunk_size=state.config.document_processing.min_chunk_size
    )
    print(f"Document processor initialized")
    
    if len(state.vector_store) > 0:
        state.document_loaded = True
        print(f"Loaded existing document with {len(state.vector_store)} chunks")
    
    print("=" * 50)
    print("RAG Pipeline Backend Ready!")
    print(f"LLM: {state.config.llm.model_name}")
    print(f"Embedding: {state.config.embeddings.model_name}")
    print(f"Hybrid Weights: BM25={state.config.hybrid_search.bm25_weight}, Semantic={state.config.hybrid_search.semantic_weight}")
    print("=" * 50)

    yield 

    print("Shutting down RAG Pipeline Backend...")




app = FastAPI(
    title="RAG Pipeline API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AppState:
    """Application state."""
    config = None
    embedding_model = None
    vector_store = None
    bm25_retriever = None
    hybrid_searcher = None
    rag_graph = None
    document_processor = None
    document_loaded = False
    current_document = None

state = AppState()

class StatusResponse(BaseModel):
    """Status response model."""
    status: str
    document_loaded: bool
    current_document: Optional[str]
    total_chunks: int




@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "embedding_model": state.config.embeddings.model_name,
        "llm_model": state.config.llm.model_name
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        status="ready" if state.document_loaded else "waiting_for_document",
        document_loaded=state.document_loaded,
        current_document=state.current_document,
        total_chunks=len(state.vector_store)
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(('.pdf', '.docx', '.doc')):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
        
        upload_dir = Path(state.config.server.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Processing: {file.filename}")
        
        chunks = state.document_processor.process_document(str(file_path))
        print(f"Extracted {len(chunks)} chunks")
        
        state.hybrid_searcher.clear()
        state.hybrid_searcher.add_documents(chunks)
        state.vector_store.save()
        state.document_loaded = True
        state.current_document = file.filename
        
        print(f"Document processed: {file.filename}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(chunks),
            "message": f"Document processed successfully with {len(chunks)} chunks"
        }
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document")
async def clear_document():
    """Clear the current document."""
    try:
        state.hybrid_searcher.clear()
        state.document_loaded = False
        state.current_document = None
        
        return {
            "status": "success",
            "message": "Document cleared successfully"
        }
        
    except Exception as e:
        print(f"Error clearing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            question_data = json.loads(data)
            question = question_data.get("question", "")
            
            if not question:
                await websocket.send_json({
                    "type": "error",
                    "data": "No question provided"
                })
                continue
            
            await websocket.send_json({
                "type": "status",
                "data": "Thinking..."
            })
            
            try:
                async for chunk in state.rag_graph.query_stream(question):
                    if chunk["type"] == "tool_start":
                        await websocket.send_json({
                            "type": "status",
                            "data": "Looking into knowledge base..."
                        })
                    
                    elif chunk["type"] == "context":
                        await websocket.send_json({
                            "type": "context",
                            "data": chunk["data"]
                        })
                    
                    elif chunk["type"] == "answer_chunk":
                        await websocket.send_json({
                            "type": "answer_chunk",
                            "data": chunk["data"]
                        })
                    
                    elif chunk["type"] == "references":
                        await websocket.send_json({
                            "type": "references",
                            "data": chunk["data"]
                        })
                    
                    elif chunk["type"] == "error":
                        await websocket.send_json({
                            "type": "error",
                            "data": chunk["data"]
                        })
                
                await websocket.send_json({
                    "type": "complete",
                    "data": "Done"
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "data": str(e)
                })
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": str(e)
            })
        except:
            pass


if __name__ == "__main__":
    print("It might take few minutes to download the Embedding model...")
    config = get_config()
    
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=False
    )
