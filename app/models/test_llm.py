from llm_engine import LLMEngine

llm = LLMEngine()
response = llm.generate(
    "qwen2.5-coder:7b", 
    "write a simple python function to add two numbers",
)

print(response)