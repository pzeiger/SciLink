from pathlib import Path
from typing import List, Dict, Any

from .pdf_parser import extract_pdf_two_pass, chunk_text
from .excel_parser import parse_adaptive_excel
from .parser_utils import get_files_from_directory


def ingest_files(file_paths: List[str], is_code_mode: bool, code_chunk_size: int = 20000, repo_name: str = None) -> List[Dict[str, Any]]:
    """
    Recursively finds files and routes them to the 
    correct parser (PDF, Excel, or Text) based on extension.
    """
    chunks = []
    expanded_paths = []
    
    # 1. Expand directories
    if file_paths:
        for f_path in file_paths:
            path_obj = Path(f_path)
            if path_obj.is_dir():
                expanded_paths.extend(get_files_from_directory(f_path))
            else:
                expanded_paths.append(f_path)

    # 2. Process each file
    for f_path in expanded_paths:
        path = Path(f_path)
        if not path.exists():
            print(f"  - ⚠️ File not found: {f_path}")
            continue
        
        file_ext = path.suffix.lower()
        
        # --- ROUTE A: PDF Documents ---
        if file_ext == '.pdf':
            pdf_chunks = extract_pdf_two_pass(f_path)
            if is_code_mode:
                for c in pdf_chunks: c['metadata']['content_type'] = 'code'
            chunks.extend(pdf_chunks)

        # --- ROUTE B: Structured Data (Excel/CSV) ---
        elif file_ext in ['.xlsx', '.xls', '.csv'] and not is_code_mode:
            print(f"  - 📊 Auto-detected Data File: {path.name}")
            potential_meta = path.with_suffix('.json')
            meta_context = str(potential_meta) if potential_meta.exists() else None
            
            try:
                data_chunks = parse_adaptive_excel(str(path), context_path=meta_context)
                chunks.extend(data_chunks)
            except Exception as e:
                print(f"    - ❌ Error parsing data file: {e}")

        # --- ROUTE C: Text & Code Files ---
        elif file_ext in ['.txt', '.md', '.py', '.java', '.r', '.cpp', '.h', '.js', '.json', '.csv']:
            try:
                with path.open('r', encoding='utf-8') as f: content = f.read()
                
                if is_code_mode:
                    formatted_text = f"CODE FILE: {path.name}\n\n```\n{content}\n```"
                    chunk_sz = code_chunk_size  # Passed as argument now
                    ctype = 'code'
                else:
                    formatted_text = f"DOCUMENT: {path.name}\n\n{content}"
                    chunk_sz = 1000
                    ctype = 'text'
                
                new_chunks = chunk_text(formatted_text, page_num=1, chunk_size=chunk_sz, overlap=50)
                for c in new_chunks: 
                    c['metadata']['content_type'] = ctype
                    c['metadata']['source'] = f_path
                    if repo_name: c['metadata']['repo_name'] = repo_name
                chunks.extend(new_chunks)
            except Exception as e:
                print(f"  - ❌ Error reading text file {f_path}: {e}")
        else:
            if not path.name.startswith('.'):
                print(f"  - ⚠️ Unsupported file type: {f_path}")

    return chunks