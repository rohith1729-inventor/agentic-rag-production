import os
import glob
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
DATA_DIR = "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 100  # Merge chunks smaller than this

def load_documents(data_dir: str) -> List[Dict]:
    """Loads text and markdown files from the data directory."""
    docs = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir} directory. Please add .txt or .md files.")
        return docs

    files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True) + \
            glob.glob(os.path.join(data_dir, "**/*.md"), recursive=True)
    
    print(f"Found {len(files)} documents.")
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            docs.append({"source": file_path, "text": text})
    return docs

def merge_small_chunks(chunks: List[str], min_size: int) -> List[str]:
    """
    Strict Rule: If chunks are very small/similar, merge them to preserve context.
    This iterates through chunks and merges a small chunk into the previous one.
    """
    merged_chunks = []
    current_chunk = ""

    for chunk in chunks:
        if len(current_chunk) + len(chunk) < min_size:
            # If combining doesn't exceed a 'safe' merge limit, or just append if current is empty
            current_chunk += "\n" + chunk if current_chunk else chunk
        elif len(chunk) < min_size:
             # Current chunk is small, append to previous (if reasonable)
             current_chunk += " " + chunk
        else:
            if current_chunk:
                merged_chunks.append(current_chunk)
            current_chunk = chunk
    
    if current_chunk:
        merged_chunks.append(current_chunk)
        
    return merged_chunks

def process_documents():
    # 1. Load Data
    docs = load_documents(DATA_DIR)
    if not docs:
        print("No documents found. Exiting.")
        return

    # 2. Initialize Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    # 3. Load Model
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    total_chunks = 0
    
    for doc in docs:
        print(f"\nProcessing: {doc['source']}")
        raw_chunks = text_splitter.split_text(doc["text"])
        
        # 4. Strict Merge Logic
        final_chunks = merge_small_chunks(raw_chunks, MIN_CHUNK_SIZE)
        
        print(f"  - Original Chunks: {len(raw_chunks)}")
        print(f"  - Merged Chunks:   {len(final_chunks)}")

        # 5. Generate Embeddings
        embeddings = model.encode(final_chunks)
        
        # 6. Print Summary (Proof of Work)
        for i, (chunk, vector) in enumerate(zip(final_chunks, embeddings)):
            print(f"    [Chunk {i}] Len: {len(chunk)} | Embedding Dim: {len(vector)}")
            # print(f"    Preview: {chunk[:50]}...") # Optional preview

        total_chunks += len(final_chunks)

    print(f"\nTotal Chunks Processed: {total_chunks}")

if __name__ == "__main__":
    process_documents()
