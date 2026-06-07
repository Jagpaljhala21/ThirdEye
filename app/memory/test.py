import time
from chromadb.utils import embedding_functions

print("[1/3] Initializing the Embedding Function...")
start_time = time.time()

# The FIRST time this line runs, it will connect to Hugging Face
# and download the ~90MB all-MiniLM-L6-v2 model weights to your local cache.
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

load_time = time.time() - start_time
print(f"[2/3] Model loaded in {load_time:.2f} seconds.")

print("[3/3] Testing vector generation...")
# Pass a sample memory through the model
sample_text = ["The user prefers text-based tutorials over video content."]
vector = embedding_func(sample_text)

# all-MiniLM-L6-v2 always outputs exactly 384 dimensions
print(f"\n[SUCCESS] Generated embedding vector with {len(vector[0])} dimensions!")