from app.memory.memory_manager import MemoryManager

# Initialize the manager
memory = MemoryManager()

# 1. Store a memory
memory.store_memory(
    original_query="Write a Python script to calculate Pi.",
    final_output="Use sympy.pi. Do not attempt to calculate an exact integer as Pi is irrational.",
    lessons_learned="Never attempt to represent Pi as a terminating integer."
)

# 2. Test an exact match retrieval
print("\n--- Testing Exact Match ---")
context = memory.retrieve_context("Write a Python script to calculate Pi.")
print(context)

# 3. Test a semantic similarity retrieval
print("\n--- Testing Semantic Match ---")
context = memory.retrieve_context("How do I get the value of Pi in Python without approximating?")
print(context)

# 4. Test a completely unrelated query (Should return empty string)
print("\n--- Testing Unrelated Query ---")
context = memory.retrieve_context("How do I cook a steak?")
if not context:
    print("Correctly ignored unrelated query.")