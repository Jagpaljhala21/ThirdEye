import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class MemoryManager:
    def __init__(self, storage_dir: str = "./memory_store"):
        """
        Initializes the FAISS vector index and parallel JSON metadata store.
        """
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(storage_dir, "metadata.json")
        
        # Ensure the storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)

        # Load the hyper-optimized local embedding model (384 dimensions)
        print("[MEMORY] Loading all-MiniLM-L6-v2 embedding model...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384 

        # Load or create the FAISS index (using L2 distance)
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"[MEMORY] Loaded existing FAISS index with {self.index.ntotal} vectors.")
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            print("[MEMORY] Created new FAISS index.")

        # Load or create the parallel metadata store
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_to_disk(self):
        """Synchronizes the FAISS index and JSON metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def store_memory(self, original_query: str, final_output: str, lessons_learned: str = ""):
        """
        Embeds the query, adds it to the FAISS index, and stores the text payload.
        """
        # 1. Generate the embedding vector
        vector = self.encoder.encode([original_query])
        # FAISS strictly requires float32 numpy arrays
        vector = np.array(vector).astype('float32') 

        # 2. Get the next available ID
        memory_id = self.index.ntotal 

        # 3. Add to FAISS index
        self.index.add(vector)

        # 4. Save the actual text data to the metadata dictionary
        self.metadata[str(memory_id)] = {
            "query": original_query,
            "output": final_output,
            "lessons": lessons_learned
        }

        # 5. Persist to disk
        self._save_to_disk()
        print(f"[MEMORY STORED] Indexed cognitive experience ID: {memory_id}")

    def retrieve_context(self, current_query: str, top_k: int = 1, distance_threshold: float = 1.0) -> str:
        """
        Searches FAISS for similar past queries. If a match is found within the 
        distance threshold, it returns the past knowledge.
        """
        if self.index.ntotal == 0:
            return ""

        # 1. Embed the incoming query
        query_vector = self.encoder.encode([current_query])
        query_vector = np.array(query_vector).astype('float32')

        # 2. Search the FAISS index
        # Returns the L2 distances and the integer IDs of the closest matches
        distances, indices = self.index.search(query_vector, top_k)

        best_distance = distances[0][0]
        best_id = str(indices[0][0])

        # 3. Evaluate the relevance (Lower L2 distance = closer match)
        if best_distance > distance_threshold or best_id not in self.metadata:
            return ""

        # 4. Extract the payload and format for the LLM
        past_data = self.metadata[best_id]
        past_output = past_data["output"]
        lessons = past_data.get("lessons", "")
        
        memory_context = (
            "\n=========================================\n"
            "SYSTEM MEMORY (PAST SIMILAR EXPERIENCE):\n"
            "The system encountered a similar task in the past and successfully resolved it.\n"
        )
        if lessons:
            memory_context += f"CRITICAL LESSON LEARNED: {lessons}\n\n"
            
        memory_context += (
            f"PAST SUCCESSFUL OUTPUT:\n{past_output}\n"
            "Use this past knowledge to inform your current response, but adapt it to the specific constraints of the new prompt.\n"
            "=========================================\n"
        )
        
        print(f"[MEMORY RETRIEVED] Found relevant context (Distance: {best_distance:.4f})")
        return memory_context