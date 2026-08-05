from pathlib import Path
from typing import Callable, Dict, List, Optional

import gc

from config import DATA_DIR, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_NAME, PDF_DIR, SUPPORTED_EXTENSIONS
from utils.chunking import chunk_pdf_pages
from utils.embeddings import embed_texts, load_embedding_model
from utils.document_parsers import extract_document_pages, extract_pdf_pages
from utils.vector_store import (
    create_empty_index,
    load_index,
    load_metadata,
    persist_vector_store,
)

ProgressCallback = Callable[[Dict[str, object]], None]


def _emit_progress(progress_callback: Optional[ProgressCallback], **event: object) -> None:
    if progress_callback:
        progress_callback(event)


def build_index_incrementally(
    metadata: List[dict],
    embedding_model_name: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> None:
    """Embed and persist metadata in small batches to avoid CPU RAM spikes."""
    if not metadata:
        raise ValueError("Cannot build a vector store without metadata.")

    _emit_progress(
        progress_callback,
        phase="loading_model",
        message=f"Loading embedding model: {embedding_model_name}",
        current=0,
        total=len(metadata),
    )
    embedding_model = load_embedding_model(embedding_model_name)
    index = None
    total_batches = (len(metadata) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    for start in range(0, len(metadata), EMBEDDING_BATCH_SIZE):
        batch = metadata[start:start + EMBEDDING_BATCH_SIZE]
        batch_number = (start // EMBEDDING_BATCH_SIZE) + 1
        _emit_progress(
            progress_callback,
            phase="embedding",
            message=f"Embedding batch {batch_number} of {total_batches}",
            current=start,
            total=len(metadata),
            batch=batch_number,
            total_batches=total_batches,
        )
        texts = [chunk["text"] for chunk in batch]
        embeddings = embed_texts(embedding_model, texts)
        if index is None:
            index = create_empty_index(embeddings.shape[1])
        index.add(embeddings)
        del embeddings
        gc.collect()
        _emit_progress(
            progress_callback,
            phase="embedding",
            message=f"Embedded {min(start + len(batch), len(metadata))} of {len(metadata)} chunks",
            current=min(start + len(batch), len(metadata)),
            total=len(metadata),
            batch=batch_number,
            total_batches=total_batches,
        )

    if index is None:
        raise ValueError("No embeddings were created.")

    _emit_progress(
        progress_callback,
        phase="persisting",
        message="Saving FAISS, metadata, and BM25 indexes",
        current=len(metadata),
        total=len(metadata),
    )
    persist_vector_store(index, metadata, embedding_model_name)
    _emit_progress(
        progress_callback,
        phase="complete",
        message="Vector store rebuilt",
        current=len(metadata),
        total=len(metadata),
    )


def append_index_incrementally(
    index,
    existing_metadata: List[dict],
    new_metadata: List[dict],
    embedding_model_name: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> None:
    """Append new chunks to an existing FAISS index in small embedding batches."""
    _emit_progress(
        progress_callback,
        phase="loading_model",
        message=f"Loading embedding model: {embedding_model_name}",
        current=0,
        total=len(new_metadata),
    )
    embedding_model = load_embedding_model(embedding_model_name)
    total_batches = (len(new_metadata) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    for start in range(0, len(new_metadata), EMBEDDING_BATCH_SIZE):
        batch = new_metadata[start:start + EMBEDDING_BATCH_SIZE]
        batch_number = (start // EMBEDDING_BATCH_SIZE) + 1
        _emit_progress(
            progress_callback,
            phase="embedding",
            message=f"Embedding batch {batch_number} of {total_batches}",
            current=start,
            total=len(new_metadata),
            batch=batch_number,
            total_batches=total_batches,
        )
        texts = [chunk["text"] for chunk in batch]
        embeddings = embed_texts(embedding_model, texts)
        index.add(embeddings)
        del embeddings
        gc.collect()
        _emit_progress(
            progress_callback,
            phase="embedding",
            message=f"Embedded {min(start + len(batch), len(new_metadata))} of {len(new_metadata)} chunks",
            current=min(start + len(batch), len(new_metadata)),
            total=len(new_metadata),
            batch=batch_number,
            total_batches=total_batches,
        )

    existing_metadata.extend(new_metadata)
    _emit_progress(
        progress_callback,
        phase="persisting",
        message="Saving FAISS, metadata, and BM25 indexes",
        current=len(new_metadata),
        total=len(new_metadata),
    )
    persist_vector_store(index, existing_metadata, embedding_model_name)
    _emit_progress(
        progress_callback,
        phase="complete",
        message="Vector store updated",
        current=len(new_metadata),
        total=len(new_metadata),
    )


def preload_embedding_model(model_name: str) -> None:
    """Download and cache the selected embedding model before ingestion."""
    load_embedding_model(model_name, local_files_only=False)


def ingest_pdf(
    pdf_path: Path,
    embedding_model_name: str = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[dict]:
    """Process a PDF, chunk it, embed chunks, and update the local FAISS store."""
    if embedding_model_name is None:
        embedding_model_name = EMBEDDING_MODEL_NAME

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    _emit_progress(
        progress_callback,
        phase="extracting",
        message=f"Extracting pages from {pdf_path.name}",
        current=0,
        total=1,
        file=pdf_path.name,
    )
    pages = extract_pdf_pages(pdf_path)
    _emit_progress(
        progress_callback,
        phase="chunking",
        message=f"Adaptive chunking {pdf_path.name}",
        current=0,
        total=len(pages),
        file=pdf_path.name,
        pages=len(pages),
    )
    chunks = chunk_pdf_pages(pages, source_filename=pdf_path.name)
    if not chunks:
        raise ValueError("No valid text chunks were extracted from the PDF.")
    _emit_progress(
        progress_callback,
        phase="chunking",
        message=f"Created {len(chunks)} chunks from {pdf_path.name}",
        current=len(pages),
        total=len(pages),
        file=pdf_path.name,
        pages=len(pages),
        chunks=len(chunks),
    )

    index = load_index()
    metadata = load_metadata()

    if index is None or index.ntotal == 0:
        build_index_incrementally(chunks, embedding_model_name, progress_callback)
    elif any(item.get("source") == pdf_path.name or item.get("document_name") == pdf_path.name for item in metadata):
        retained_metadata = [
            item for item in metadata
            if item.get("source") != pdf_path.name and item.get("document_name") != pdf_path.name
        ]
        rebuilt_metadata = retained_metadata + chunks
        build_index_incrementally(rebuilt_metadata, embedding_model_name, progress_callback)
    else:
        append_index_incrementally(index, metadata, chunks, embedding_model_name, progress_callback)

    return chunks


def ingest_pdf_folder(
    pdf_dir: Path = DATA_DIR,
    embedding_model_name: str = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, object]:
    """Rebuild the vector store using all supported documents currently present in data directory."""
    if embedding_model_name is None:
        from config import EMBEDDING_MODEL_NAME
        embedding_model_name = EMBEDDING_MODEL_NAME

    doc_paths: List[Path] = []
    pdf_dir_path = Path(pdf_dir)

    # Search root data directory and any subdirectories (e.g. data/pdfs)
    for ext in SUPPORTED_EXTENSIONS:
        doc_paths.extend(pdf_dir_path.glob(f"*{ext}"))
        doc_paths.extend(pdf_dir_path.glob(f"*{ext.upper()}"))
        doc_paths.extend(pdf_dir_path.glob(f"**/*{ext}"))
        doc_paths.extend(pdf_dir_path.glob(f"**/*{ext.upper()}"))

    doc_paths = sorted(list(set(doc_paths)))

    if not doc_paths:
        raise FileNotFoundError(f"No supported document files were found in: {pdf_dir}")

    all_chunks: List[dict] = []
    files = []
    for file_index, doc_path in enumerate(doc_paths, start=1):
        _emit_progress(
            progress_callback,
            phase="extracting",
            message=f"Extracting content from {doc_path.name} ({file_index}/{len(doc_paths)})",
            current=file_index - 1,
            total=len(doc_paths),
            file=doc_path.name,
        )
        pages = extract_document_pages(doc_path)
        _emit_progress(
            progress_callback,
            phase="chunking",
            message=f"Adaptive chunking {doc_path.name}",
            current=file_index - 1,
            total=len(doc_paths),
            file=doc_path.name,
            pages=len(pages),
        )
        chunks = chunk_pdf_pages(pages, source_filename=doc_path.name)
        if not chunks:
            files.append({
                "file": doc_path.name,
                "pages": len(pages),
                "chunks": 0,
                "status": "skipped",
            })
            continue

        all_chunks.extend(chunks)
        _emit_progress(
            progress_callback,
            phase="chunking",
            message=f"Created {len(chunks)} chunks from {doc_path.name}",
            current=file_index,
            total=len(doc_paths),
            file=doc_path.name,
            pages=len(pages),
            chunks=len(chunks),
            total_chunks=len(all_chunks),
        )
        files.append({
            "file": doc_path.name,
            "pages": len(pages),
            "chunks": len(chunks),
            "status": "processed",
        })

    if not all_chunks:
        raise ValueError("No valid text chunks were extracted from documents in data folder.")

    build_index_incrementally(all_chunks, embedding_model_name, progress_callback)
    return {
        "pdf_count": len(doc_paths),
        "chunk_count": len(all_chunks),
        "files": files,
    }


def ingest_all_documents(
    data_dir: Path = DATA_DIR,
    embedding_model_name: str = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[dict]:
    """Rebuild vector store from all supported documents in data directory and return metadata."""
    res = ingest_pdf_folder(pdf_dir=data_dir, embedding_model_name=embedding_model_name, progress_callback=progress_callback)
    return load_metadata()


def save_uploaded_pdf(uploaded_file) -> Path:
    """Save the uploaded file into the project data directory."""
    destination = DATA_DIR / uploaded_file.name
    with open(destination, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return destination
