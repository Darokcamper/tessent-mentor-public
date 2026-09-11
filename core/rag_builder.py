import os
import re
import fitz
import pickle
import subprocess
import numpy as np
from pathlib import Path

import sys

# Prevent high CPU/RAM usage and deadlocks on Windows/Intel Core i5:
# - In Streamlit: limit to 1 thread to avoid deadlocks.
# - Offline Indexing: limit to 2 threads to prevent 100% CPU lockup and thermal/RAM throttling.
is_streamlit = any("streamlit" in arg or "streamlit" in sys.argv[0].lower() for arg in sys.argv) or os.environ.get("STREAMLIT_SERVER_PORT") is not None

os.environ["OMP_NUM_THREADS"] = "1" if is_streamlit else "2"
os.environ["MKL_NUM_THREADS"] = "1" if is_streamlit else "2"
os.environ["MKL_DYNAMIC"] = "FALSE"

try:
    import torch
    torch.set_num_threads(1 if is_streamlit else 2)
except Exception:
    pass

from sentence_transformers import SentenceTransformer
import faiss


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
TXT_DIR = KNOWLEDGE_DIR / "txt"
VECTORSTORE_DIR = KNOWLEDGE_DIR / "vectorstore"

TXT_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

def ocr_document(pdf_path, ocr_pdf_path):
    """Runs ocrmypdf via subprocess on the pdf file."""
    ocrmypdf_exe = PROJECT_ROOT / "venv" / "Scripts" / "ocrmypdf.exe"
    if not os.path.exists(ocrmypdf_exe):
        ocrmypdf_exe = "ocrmypdf" # fallback to PATH
        
    print(f"Running ocrmypdf on {pdf_path.name}...")
    try:
        result = subprocess.run([
            ocrmypdf_exe,
            "--force-ocr",
            str(pdf_path),
            str(ocr_pdf_path)
        ], capture_output=True, text=True, check=True)
        print(f"OCR successful for {pdf_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ocrmypdf failed for {pdf_path.name}: {e.stderr}")
        return False

def extract_text_from_pdfs():
    """Extracts text page-by-page from all PDFs in the knowledge directory, saving them to TXT_DIR."""
    pdf_files = [p for p in KNOWLEDGE_DIR.rglob("*.pdf") if "ocr_pdfs" not in p.parts]
    print(f"Found {len(pdf_files)} PDFs to process.")
    
    # Load sources.csv catalog
    import csv
    catalog = {}
    csv_path = KNOWLEDGE_DIR / "sources.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_file = row["File"].replace("\\", "/").strip()
                catalog[rel_file] = row
                
    for pdf_path in pdf_files:
        txt_path = TXT_DIR / f"{pdf_path.stem}.txt"
        
        # Check if already cached
        if txt_path.exists() and txt_path.stat().st_size > 50:
            print(f"Already cached text for: {pdf_path.name}")
            continue
            
        rel_path = pdf_path.relative_to(KNOWLEDGE_DIR).as_posix()
        row = catalog.get(rel_path)
        is_vlsiguru = row and row.get("Source") == "VLSIGuru"
        
        if is_vlsiguru:
            print(f"Skipping VLSIGuru scanned PDF: {pdf_path.name} (will be transcribed via Gemini OCR)")
            continue
            
        print(f"Processing digital book: {pdf_path.name}...")
        try:
            doc = fitz.open(pdf_path)
            all_page_texts = []
            for page_num in range(len(doc)):
                page_text = doc[page_num].get_text("text").strip()
                all_page_texts.append(f"--- PAGE {page_num + 1} ---\n{page_text}\n")
            doc.close()
            
            # Write to cache
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_page_texts))
            print(f"Successfully cached text for digital book: {pdf_path.name}")
        except Exception as e:
            print(f"Failed to process digital book {pdf_path.name}: {e}")

def chunk_text(text, source_name, chunk_size=1000, overlap=200):
    """Splits a single PDF's cached text into chunks, keeping track of page numbers from markers."""
    chunks = []
    pages = re.split(r"--- PAGE (\d+) ---", text)
    
    if pages and not pages[0].strip().isdigit() and pages[0].strip():
        p_text = pages[0]
        chunks.extend(_make_chunks(p_text, source_name, 1, chunk_size, overlap))
        
    for i in range(1, len(pages), 2):
        try:
            current_page = int(pages[i])
        except ValueError:
            current_page = 1
            
        page_text = pages[i+1] if i+1 < len(pages) else ""
        chunks.extend(_make_chunks(page_text, source_name, current_page, chunk_size, overlap))
        
    return chunks

def _make_chunks(text, source_name, page_num, chunk_size, overlap):
    chunks = []
    text = text.strip()
    if not text:
        return []
        
    words = text.split()
    if not words:
        return []
        
    current_chunk_words = []
    current_len = 0
    
    for word in words:
        current_chunk_words.append(word)
        current_len += len(word) + 1
        
        if current_len >= chunk_size:
            chunk_text = " ".join(current_chunk_words)
            chunks.append({
                "source": source_name,
                "page": page_num,
                "text": chunk_text
            })
            
            overlap_words = []
            overlap_len = 0
            for w in reversed(current_chunk_words):
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1
                if overlap_len >= overlap:
                    break
            current_chunk_words = overlap_words
            current_len = overlap_len
            
    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        chunks.append({
            "source": source_name,
            "page": page_num,
            "text": chunk_text
        })
        
    return chunks

def build_index():
    """Reads all cached txt files, chunks them, embeds them, and builds a FAISS index."""
    txt_files = list(TXT_DIR.glob("*.txt"))
    if not txt_files:
        print("No cached text files found. Run extract_text_from_pdfs() first.")
        return
        
    print(f"Found {len(txt_files)} text files. Starting chunking...")
    all_chunks = []
    for txt_path in txt_files:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, txt_path.name)
        all_chunks.extend(chunks)
        
    print(f"Total chunks created: {len(all_chunks)}")
    if not all_chunks:
        print("No chunks to index.")
        return
        
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    print("Computing embeddings (batch_size=32)...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    
    faiss.normalize_L2(embeddings)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    index_path = str(VECTORSTORE_DIR / "faiss_index.index")
    faiss.write_index(index, index_path)
    
    metadata_path = str(VECTORSTORE_DIR / "metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(all_chunks, f)
        
    print(f"FAISS index built and saved to {index_path}")
    print(f"Metadata saved to {metadata_path}")

# --- RAG RUNTIME ---

_loaded_model = None
_loaded_index = None
_loaded_metadata = None

def _init_sentence_transformer():
    """Load SentenceTransformer from local cache first to avoid HuggingFace Hub network checks."""
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    except Exception:
        hf_token = os.getenv("HF_TOKEN", "")
        if hf_token:
            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=hf_token)
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Eagerly load the model in the main thread during module import
try:
    _loaded_model = _init_sentence_transformer()
except Exception as e:
    print(f"Warning: Loading of SentenceTransformer failed: {e}")
    _loaded_model = None

def load_rag():
    global _loaded_model, _loaded_index, _loaded_metadata
    
    index_path = VECTORSTORE_DIR / "faiss_index.index"
    metadata_path = VECTORSTORE_DIR / "metadata.pkl"
    
    if not index_path.exists() or not metadata_path.exists():
        return False
        
    if _loaded_model is None:
        try:
            _loaded_model = _init_sentence_transformer()
        except Exception as e:
            print(f"Error loading SentenceTransformer: {e}")
            return False
        
    if _loaded_index is None:
        _loaded_index = faiss.read_index(str(index_path))
        
    if _loaded_metadata is None:
        with open(metadata_path, "rb") as f:
            _loaded_metadata = pickle.load(f)
            
    return True

def search_txt_files_fallback(query, top_k=5):
    """Fallback keyword search over cached .txt files in knowledge/txt when FAISS index is building."""
    query_terms = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", query) if w.lower() not in ["what", "does", "this", "that", "command", "explain", "from", "with", "options"]]
    if not query_terms:
        return []
        
    results = []
    txt_files = list(TXT_DIR.glob("*.txt"))
    for txt_path in txt_files:
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            pages = re.split(r"--- PAGE (\d+) ---\n", content)
            for i in range(1, len(pages), 2):
                page_num = int(pages[i])
                page_text = pages[i+1]
                matches = sum(1 for term in query_terms if term in page_text.lower())
                if matches > 0:
                    results.append({
                        "source": txt_path.name,
                        "page": page_num,
                        "text": page_text[:1500],
                        "score": float(matches)
                    })
        except Exception:
            pass
            
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def retrieve_command_definition(command_name: str, source_filter: str = None, max_chars: int = 7000):
    """
    Command-dictionary-aware retrieval.

    Scans the cached page text (knowledge/txt/*.txt) for the given Tessent command name
    and returns the CONTIGUOUS manual block for that command (heading through to the next
    command heading or page boundary), so the full Usage/Arguments list arrives as one piece.

    This avoids the fragmentation problem where a single command entry is split across
    multiple ~1000-char RAG chunks, which caused the model to see only some options.

    Returns a list of dicts: {source, page, text}.
    """
    if not command_name:
        return []
    name_lower = command_name.strip().lower()
    results = []

    txt_files = sorted(TXT_DIR.glob("*.txt"))
    for txt_path in txt_files:
        if source_filter:
            src_lower = txt_path.name.lower()
            sf = source_filter.lower()
            # match if the filter text appears in the filename
            if sf not in src_lower and sf.strip(".txt").lower() not in src_lower:
                continue
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Map absolute character offsets to physical page numbers using the
        # page markers emitted by the extractor ("--- PAGE N ---").
        page_spans = []  # (start_offset_of_page_text, page_num)
        for pm in re.finditer(r"--- PAGE (\d+) ---\n?", text):
            page_spans.append((pm.end(), int(pm.group(1))))

        # Find the command heading in the whole file (so the captured block can
        # extend across page boundaries instead of stopping at the page edge).
        for m in re.finditer(re.escape(name_lower), text.lower()):
            # Must start its own line (command dictionary headings do).
            line_start = text.rfind("\n", 0, m.start()) + 1
            if text[line_start:m.start()].strip():
                continue
            # Real entries are immediately followed by "Context:" or "Note -";
            # this skips mid-sentence mentions and TOC-like references.
            nxt = text[m.end():].lstrip(" \t\r\n")
            if not nxt.startswith("Context:") and not nxt.startswith("Note -"):
                continue

            block_start = m.start()
            block_end = block_start + max_chars

            # Emit the contiguous block in per-page pieces so every piece keeps
            # the correct page citation, even when the entry spans 2+ pages.
            for i, (p_start, p_num) in enumerate(page_spans):
                p_end = page_spans[i + 1][0] if i + 1 < len(page_spans) else len(text)
                seg_start = max(block_start, p_start)
                seg_end = min(block_end, p_end)
                if seg_start >= seg_end:
                    continue
                seg = text[seg_start:seg_end].strip()
                if seg:
                    results.append({
                        "source": txt_path.name,
                        "page": p_num,
                        "text": seg
                    })
            break  # first real command-dictionary entry per file is enough
    return results

def retrieve(query, top_k=5, source_filter=None):
    """Retrieves top_k chunks matching the query, returning a list of dictionaries with source, page, text, and score."""
    # If the query contains a Tessent command name (identifiers using underscores like
    # set_failure_mapping_options, add_clocks, create_patterns, ...), prefer the precise
    # command-dictionary block so the FULL Usage/option list is returned as one contiguous
    # excerpt instead of fragmented 1000-char chunks.
    # Require at least one underscore + a real command root to avoid grabbing plain words
    # like "set", "add", "get", "run" that appear in ordinary English queries.
    cmd_hits = re.findall(r"\b(?:set|add|create|remove|delete|write|read|report|get|put|check|run|save|do|exit|source|define)_[a-z0-9_]+\b", query)
    if cmd_hits:
        # Only use precise lookup when the query is dominated by command name(s), to avoid
        # hijacking normal questions.
        defs = []
        for tok in dict.fromkeys(cmd_hits):
            defs.extend(retrieve_command_definition(tok, source_filter=source_filter))
        if defs:
            # Prefer definitions in the Shell Reference manual for the fuller option list
            defs.sort(key=lambda d: (0 if "tshell_ref" in d["source"].lower() else 1, d["page"]))
            return defs[:top_k]
    if not load_rag():
        return search_txt_files_fallback(query, top_k=top_k)
        
    query_vector = _loaded_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)
    
    # If source filter is specified, perform exact local cosine similarity ranking on chunks of that specific file
    if source_filter:
        sf_clean = source_filter.lower().replace(".txt", "").replace(".pdf", "")
        doc_indices = []
        doc_chunks = []
        
        for idx, chunk in enumerate(_loaded_metadata):
            chunk_src_clean = chunk["source"].lower().replace(".txt", "").replace(".pdf", "")
            if sf_clean in chunk_src_clean:
                doc_indices.append(idx)
                doc_chunks.append(chunk)
                
        if not doc_chunks:
            return []
            
        # Reconstruct vectors from FAISS index and calculate cosine similarities locally
        import numpy as np
        doc_vectors = []
        for idx in doc_indices:
            vec = _loaded_index.reconstruct(idx)
            doc_vectors.append(vec)
            
        doc_vectors = np.array(doc_vectors).astype("float32")
        faiss.normalize_L2(doc_vectors)
        
        # Dot product of normalized vectors gives cosine similarity
        scores = np.dot(doc_vectors, query_vector[0])
        
        # Boost scores based on exact keyword/number matching to handle follow-up queries like "what about 3 and 4"
        boosted_ranked = []
        for score, chunk in zip(scores, doc_chunks):
            boost = 0.0
            # Check for numbers in the query (e.g. "3", "4") matching rule headers (e.g. "3.", "4.")
            for num in re.findall(r"\b\d+\b", query):
                # Strong boost for slide headers starting with '# 3' or '# 4' or starting with '3.'
                if re.search(r"#\s*" + num + r"\b", chunk["text"]):
                    boost += 1.2
                elif re.search(r"^\s*" + num + r"\b[\.\-:]", chunk["text"]):
                    boost += 1.2
                # Medium boost for rule numbers inside the text
                elif re.search(r"\b" + num + r"\b[\.\-:]", chunk["text"]):
                    boost += 0.4
                elif re.search(r"page\s+" + num + r"\b", chunk["text"].lower()):
                    boost += 0.4
                    
            # Check for specific technical terms in the query matching chunk text
            query_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", query) if w.lower() not in ["what", "about", "from", "scan", "rule", "rules", "with", "this", "that", "they", "them"]]
            for qw in query_words:
                if qw in chunk["text"].lower():
                    boost += 0.1
                    
            boosted_ranked.append((score + boost, chunk))
            
        ranked = sorted(boosted_ranked, key=lambda x: x[0], reverse=True)
        
        # Retrieve the top_k boosted results
        top_results = ranked[:top_k]
        
        # Sort the top results by page number to keep chronological slide sequence for the LLM
        sorted_results = sorted(top_results, key=lambda x: x[1]["page"])
        
        results = []
        for score, chunk in sorted_results:
            results.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score)
            })
        return results

    # Global search when no source filter is specified
    scores, indices = _loaded_index.search(query_vector, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_loaded_metadata):
            continue
        chunk = _loaded_metadata[idx]
        results.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": float(score)
        })
        
    return results

def upload_and_index_pdf(file_bytes, filename, topic="Digital Book"):
    """
    Saves an uploaded PDF to knowledge/, extracts text page-by-page, updates sources.csv,
    rebuilds the FAISS vector index, and reloads RAG runtime memory.
    """
    global _loaded_index, _loaded_metadata
    
    topic_folder_map = {
        "Shell User Manual": "01_Tessent_Shell_User_Manual",
        "Shell Reference Manual": "02_Tessent_Shell_Reference_Manual",
        "ATPG & Scan User Manual": "03_Scan_and_ATPG_User_Manual",
        "Library User Manual": "04_Library_User_Manual",
        "Scan & ATPG Lab Manual": "05_Scan_and_ATPG_Lab_Manual",
        "SCAN": "03_Scan_and_ATPG_User_Manual",
        "ATPG": "03_Scan_and_ATPG_User_Manual",
        "EDT": "03_Scan_and_ATPG_User_Manual",
        "LINUX": "01_Tessent_Shell_User_Manual",
        "TCL": "02_Tessent_Shell_Reference_Manual"
    }
    target_dir = KNOWLEDGE_DIR / topic_folder_map.get(topic, "01_Tessent_Shell_User_Manual")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = target_dir / filename
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)
        
    csv_path = KNOWLEDGE_DIR / "sources.csv"
    rel_path = pdf_path.relative_to(KNOWLEDGE_DIR).as_posix()
    if csv_path.exists():
        import csv
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([999, rel_path, topic, "Digital Book"])
            
    txt_path = TXT_DIR / f"{pdf_path.stem}.txt"
    try:
        doc = fitz.open(pdf_path)
        all_page_texts = []
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text("text").strip()
            all_page_texts.append(f"--- PAGE {page_num + 1} ---\n{page_text}\n")
        doc.close()
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_page_texts))
    except Exception as e:
        return f"Error extracting text from PDF: {e}"
        
    build_index()
    
    _loaded_index = None
    _loaded_metadata = None
    load_rag()
    
    return f"Successfully uploaded '{filename}', extracted {len(all_page_texts)} pages, and updated FAISS vector index!"
def index_text_file(file_bytes, filename, topic="Digital Book"):
    """Index a non-PDF text document (DOCX/PPTX/TXT) into the RAG knowledge base.

    Mirrors upload_and_index_pdf: saves a copy under knowledge/, writes its paged
    text to knowledge/txt/, records it in sources.csv, rebuilds the FAISS index,
    and reloads RAG runtime memory. Returns a status string.
    """
    global _loaded_index, _loaded_metadata

    from core.file_reader import extract_document_text

    # Sanitize the stored filename to avoid path traversal / collisions.
    safe_name = Path(filename).name
    stem = Path(safe_name).stem
    ext_suffix = Path(safe_name).suffix.lower()

    # Preserve existing '--- PAGE N ---' markers for .txt when present.
    pages = extract_document_text(file_bytes, safe_name)

    # Save a copy under knowledge (root "Uploads" folder) for provenance.
    uploads_dir = KNOWLEDGE_DIR / "Uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    stored_path = uploads_dir / safe_name
    with open(stored_path, "wb") as f:
        f.write(file_bytes)

    # Write paged text to the txt cache (this is what build_index() consumes).
    txt_path = TXT_DIR / f"{stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(pages))

    # Record in sources.csv for provenance/traceability.
    csv_path = KNOWLEDGE_DIR / "sources.csv"
    rel_path = stored_path.relative_to(KNOWLEDGE_DIR).as_posix()
    try:
        import csv
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([999, rel_path, topic, "User Upload"])
    except Exception:
        pass

    build_index()

    _loaded_index = None
    _loaded_metadata = None
    load_rag()

    return f"Successfully uploaded '{safe_name}', extracted {len(pages)} page(s), and updated FAISS vector index!"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        print("Extracting text from PDFs...")
        extract_text_from_pdfs()
        print("Building index...")
        build_index()
    elif len(sys.argv) > 1 and sys.argv[1] == "index":
        print("Building index from current cached text files...")
        build_index()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) > 2:
            query = sys.argv[2]
        else:
            query = "lockup latch hold violation"
        print(f"Testing retrieval for: '{query}'")
        res = retrieve(query)
        for i, r in enumerate(res):
            print(f"\n[{i+1}] Score: {r['score']:.4f} | Source: {r['source']} (Page {r['page']})")
            print("-" * 60)
            print(r['text'])
