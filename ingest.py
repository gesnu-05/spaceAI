import shutil
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding

PERSIST_DIR = "./storage"
DATA_DIR = "./data"

shutil.rmtree(PERSIST_DIR, ignore_errors=True)

# ✅ Use Ollama for local embeddings
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

documents = SimpleDirectoryReader(DATA_DIR).load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
index.storage_context.persist(persist_dir=PERSIST_DIR)

print("✅ Index built using Ollama and saved to storage/")
