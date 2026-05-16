import logging
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    documents = load_all_documents("data")
    logging.info("Documents loaded:")
    logging.info(len(documents))
    logging.info("Chunking documents..")
    chunks = EmbeddingPipeline().chunk_documents(documents)
    logging.info("Creating Embeddings")
    vectors = EmbeddingPipeline().embed_chunks(chunks=chunks)
    logging.info("Embeddings created")
    logging.info(len(vectors))