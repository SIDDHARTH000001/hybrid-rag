from typing import TypedDict, List, Dict, Any, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from backend.hybrid_search import HybridSearcher
from backend.document_processor import DocumentChunk
from pydantic import BaseModel, Field
import json


class RAGGraph:
    
    def __init__(
        self,
        hybrid_searcher: HybridSearcher,
        model_name: str = "gemini-2.5-flash",
        provider: str = "gcp",
        endPoint: str = "",
        version: str = "",
        api_key: str = "2025-01-01-preview",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_k: int = 5
    ):
        self.hybrid_searcher = hybrid_searcher
        self.provider = provider  
        
        if provider == "gcp":
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        elif provider == "azure":
            self.llm = AzureChatOpenAI(
                temperature=temperature,
                azure_endpoint=endPoint,
                api_key=api_key,
                deployment_name=model_name,
                api_version=version,
                max_tokens=max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.search_tool = self._create_search_tool(top_k)
        self.agent = self._create_agent()
    
    def _create_search_tool(self, top_k: int):
        @tool
        def search_document(query: str, top_k: int = top_k) -> str:
            """Search the uploaded document for relevant information.
            
            Use this tool ONLY when the user asks about the uploaded document content.
            Do not use for general knowledge questions.
            
            Args:
                query: What to search for in the document
                
            Returns:
                Relevant text from the document with source references
            """
            if len(self.hybrid_searcher.vector_store) == 0:
                return "ERROR: No document has been uploaded. Tell the user to upload a document first."
            
            results = self.hybrid_searcher.search(query, top_k=top_k)
            
            if not results:
                return "The provided document does not contain information about this topic."
            
            context_parts = []
            for i, (chunk, score, score_details) in enumerate(results, 1):
                context_parts.append(
                    f"[Source {i}]\n"
                    f"Reference: {chunk.file_name}, Page {chunk.page_number}, "
                    f"Line {chunk.line_number}, Chunk #{chunk.chunk_id}"
                    + (f", Section: {chunk.section}" if chunk.section else "") + "\n"
                    f"Content: {chunk.text}\n"
                )
            
            return "\n".join(context_parts)
        
        return search_document
    
    def _create_agent(self):
        self.system_message = """You are an AI assistant whose ONLY source of factual answers is the provided document knowledge base. For every user question:

- First, check the document content and ONLY use information found in the document search results.
- If the answer can be found in the document, answer using the relevant sections and cite source references clearly.
- If the answer cannot be found in the document content, do NOT provide answers from your general knowledge or training.
- Any question which can which is not realted to document centext should be replied exactly with: “The provided document does not contain this information.”

Never use your general knowledge outside the document. Always prioritize evidence from the document search tool results, and only respond based on what is retrieved.

if user query found in document then response strictly in this format :

    "The answer to the question"
    "references: All relevant references list for the answer, e.g 'filename, Page No, Line No, Chunk id'"

"""
        
        agent = create_react_agent(
            self.llm,
            tools=[self.search_tool]
        )
        
        return agent
    
    def query(self, question: str) -> Dict[str, Any]:

        result = self.agent.invoke({
            "messages": [
                SystemMessage(content=self.system_message),
                HumanMessage(content=question)
            ]
        })
        
        messages = result.get("messages", [])
        answer = ""
        if messages:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    answer = self._extract_content(msg)
                    break
        
        references = self._extract_references(messages)
        
        return {
            "answer": answer,
            "references": references,
            "retrieved_chunks": []
        }
    
    def _extract_content(self, msg: AIMessage) -> str:
        content = msg.content
        
        # Handle list format (Gemini)
        if isinstance(content, list):
            if len(content) > 0:
                if isinstance(content[0], dict) and "text" in content[0]:
                    return content[0]["text"]
                elif isinstance(content[0], str):
                    return content[0]
            return ""
        
        # Handle string format (Azure)
        if isinstance(content, str):
            return content
        
        return str(content)
    
    async def query_stream(self, question: str):
        tool_called = False
        tool_output = ""
        full_answer = ""
        references = []
        
        try:
            async for event in self.agent.astream_events(
                {
                    "messages": [
                        SystemMessage(content=self.system_message),
                        HumanMessage(content=question)
                    ]
                },
                version="v2" 
            ):
                kind = event["event"]
            
                if kind == "on_tool_start":
                    tool_called = True
                    yield {
                        "type": "tool_start",
                        "data": "Searching document..."
                    }
                
                elif kind == "on_tool_end":
                    output = event.get("data", {}).get("output", "")
                    
                    if hasattr(output, 'content'):
                        tool_output = output.content
                    else:
                        tool_output = str(output)
                    
                    refs = self._parse_references_from_tool_output(tool_output)
                    if refs:
                        references = refs
                        yield {
                            "type": "context",
                            "data": {
                                "num_chunks": len(refs),
                                "message": f"Found {len(refs)} relevant sections"
                            }
                        }
                
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if not chunk:
                        continue
                    
                    content = self._extract_streaming_content(chunk)
                    
                    if content:
                        full_answer += content
                        yield {
                            "type": "answer_chunk",
                            "data": content
                        }
            
            if not full_answer:
                full_answer = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                yield {
                    "type": "answer_chunk",
                    "data": full_answer
                }
            
            yield {
                "type": "references",
                "data": references if tool_called else []
            }
        
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"Streaming error: {error_msg}")
            yield {
                "type": "error",
                "data": error_msg
            }
            yield {
                "type": "references",
                "data": []
            }
    
    def _extract_streaming_content(self, chunk):
        if not chunk:
            return ""
        
        content = getattr(chunk, 'content', None)
        
        if content is None:
            return ""
        
        if isinstance(content, list):
            if len(content) == 0:
                return ""
            
            first_item = content[0]
            if isinstance(first_item, dict):
                return first_item.get("text", "")
            elif isinstance(first_item, str):
                return first_item
            return ""
        
        if isinstance(content, str):
            return content
        
        return ""
    
    def _extract_references(self, messages: List[BaseMessage]):
        references = []
        
        for msg in messages:
            if isinstance(msg, ToolMessage):
                refs = self._parse_references_from_tool_output(msg.content)
                references.extend(refs)
            
            elif isinstance(msg, AIMessage):
                if hasattr(msg, 'additional_kwargs'):
                    tool_calls = msg.additional_kwargs.get('tool_calls', [])
            
            elif hasattr(msg, 'content') and isinstance(msg.content, str):
                if 'Reference:' in msg.content:
                    refs = self._parse_references_from_tool_output(msg.content)
                    references.extend(refs)
        
        return list(dict.fromkeys(references))
    
    def _parse_references_from_tool_output(self, tool_output: str) -> List[str]:
        references = []
        
        if not isinstance(tool_output, str):
            if hasattr(tool_output, 'content'):
                tool_output = tool_output.content
            else:
                tool_output = str(tool_output)
        
        lines = tool_output.split('\n')
        for line in lines:
            if 'Reference:' in line:
                ref = line.split('Reference:')[1].strip()
                ref = ref.split('\n')[0].strip()
                if ref:
                    references.append(f"📄 {ref}")
        
        return references