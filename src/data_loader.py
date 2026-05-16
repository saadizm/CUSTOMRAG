import os
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import logging

logger = logging.getLogger(__name__)



def load_all_documents(data_dir:str) -> List[Any]:

    data_path = Path(data_dir).resolve()
    logger.info(f"[DEBUG] Data path: {data_path}")
    documents = []

    pdf_files = list(data_path.glob('**/*.pdf'))
    logger.info(f"Found {len(pdf_files)} PDF file(s) in {data_path}")

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} page(s) from {pdf_file.name}")
        except Exception as e:
            logger.error(f"Failed to load {pdf_file.name}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents
