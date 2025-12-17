import fitz  # PyMuPDF
from docx import Document
from typing import List, Dict, Any
from pathlib import Path
import re


class DocumentChunk:
    def __init__(
        self,
        text: str,
        chunk_id: int,
        page_number: int,
        line_number: int,
        section: str = "",
        file_name: str = ""
    ):

        self.text = text
        self.chunk_id = chunk_id
        self.page_number = page_number
        self.line_number = line_number
        self.section = section
        self.file_name = file_name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "line_number": self.line_number,
            "section": self.section,
            "file_name": self.file_name
        }
    
    def get_reference(self) -> str:
        ref = f"Page {self.page_number}, Line {self.line_number}, Chunk #{self.chunk_id}"
        if self.section:
            ref = f"{ref} (Section: {self.section})"
        return ref


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, min_chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def process_pdf(self, file_path: str):
        chunks = []
        chunk_id = 0
        file_name = Path(file_path).name
        doc = fitz.open(file_path)
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            
            lines = text.split('\n')
            current_text = ""
            current_line = 1
            section = self._extract_section(lines[:5])
            
            for line_idx, line in enumerate(lines, start=1):
                current_text += line + "\n"
                
                if len(current_text) >= self.chunk_size:
                    chunk_text = current_text.strip()
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(DocumentChunk(
                            text=chunk_text,
                            chunk_id=chunk_id,
                            page_number=page_num,
                            line_number=current_line,
                            section=section,
                            file_name=file_name
                        ))
                        chunk_id += 1
                    
                    overlap_text = current_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_text = overlap_text
                    current_line = line_idx
            
            if current_text.strip() and len(current_text.strip()) >= self.min_chunk_size:
                chunks.append(DocumentChunk(
                    text=current_text.strip(),
                    chunk_id=chunk_id,
                    page_number=page_num,
                    line_number=current_line,
                    section=section,
                    file_name=file_name
                ))
                chunk_id += 1
        
        doc.close()
        return chunks
    
    def process_docx(self, file_path: str):

        chunks = []
        chunk_id = 0
        file_name = Path(file_path).name
        
        doc = Document(file_path)
        
        current_text = ""
        current_page = 1  # Approximate page (incremented every ~40 paragraphs)
        current_line = 1
        section = ""
        paragraph_count = 0
        
        for para_idx, paragraph in enumerate(doc.paragraphs, start=1):
            text = paragraph.text.strip()
            
            if not text:
                continue
            
            if paragraph.style.name.startswith('Heading'):
                section = text
            
            current_text += text + "\n"
            paragraph_count += 1
            
            if paragraph_count % 20 == 0:
                current_page += 1
            
            if len(current_text) >= self.chunk_size:
                chunk_text = current_text.strip()
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        page_number=current_page,
                        line_number=current_line,
                        section=section,
                        file_name=file_name
                    ))
                    chunk_id += 1

                overlap_text = current_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_text = overlap_text
                current_line = para_idx
        
        if current_text.strip() and len(current_text.strip()) >= self.min_chunk_size:
            chunks.append(DocumentChunk(
                text=current_text.strip(),
                chunk_id=chunk_id,
                page_number=current_page,
                line_number=current_line,
                section=section,
                file_name=file_name
            ))
            chunk_id += 1
        
        return chunks
    
    def process_document(self, file_path: str):

        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == '.pdf':
            return self.process_pdf(file_path)
        elif extension in ['.docx', '.doc']:
            return self.process_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _extract_section(self, lines: List[str]) -> str:
        for line in lines:
            line = line.strip()
            if line and len(line) < 100 and not line.endswith('.'):
                if line.isupper() or (line[0].isupper() and sum(1 for c in line if c.isupper()) > len(line) * 0.3):
                    return line
        return ""
