'''
import ollama
import subprocess

class LLMEngine:
    def generate(self,model_name,prompt):
        response = ollama.chat(
            model = model_name,
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        subprocess.run(
            ["ollama","stop",model_name],
            capture_output=True,
        )
        return response['message']['content']
        
'''
import os
import time
import groq
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class LLMEngine:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # The Master Queue (Immutable blueprint)
        self.master_queue = [
            "llama-3.3-70b-versatile",       
            "openai/gpt-oss-120b",
            "meta-llama/llama-4-scout-17b-16e-instruct",         
            "qwen/qwen3-32b",                
            "openai/gpt-oss-20b"             
        ]
        
        # --- THE STATE-AWARE QUEUE ---
        # This list will dynamically shrink if models suffer permanent failures.
        self.active_queue = self.master_queue.copy()

    def generate(
        self,
        model,
        prompt,
        max_completion_tokens=1000  # Bumped to 1000 so your reasoning nodes don't cut off!
    ):
        # We copy the ACTIVE queue, not the master queue. 
        # If a model was banned on a previous pass, it won't even be in this list.
        execution_queue = self.active_queue.copy()
        
        if model:
            if model in execution_queue:
                execution_queue.remove(model)
            execution_queue.insert(0, model)

        for attempt, current_model in enumerate(execution_queue):
            try:
                response = self.client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=max_completion_tokens
                )
                
                if response.choices[0].finish_reason == "length":
                    print(f"    [WARNING] {current_model} hit the output token ceiling.")
                    
                return response.choices[0].message.content

            # 1. TRANSIENT ERRORS (Keep model in active_queue)
            except groq.RateLimitError as e:
                print(f"    [HTTP 429] Rate Limit hit on {current_model}. Rerouting...")
                time.sleep(1) 

            except (groq.InternalServerError, groq.APIConnectionError) as e:
                print(f"    [HTTP 50X] Server error on {current_model}. Rerouting...")
                
            # 2. PERSISTENT ERRORS (Ban model from active_queue)
            except groq.BadRequestError as e:
                error_msg = str(e).lower()
                
                if "context" in error_msg or "tokens" in error_msg:
                    print(f"    [STATE UPDATE] Context overflow on {current_model}. Permanently removing from router.")
                    if current_model in self.active_queue:
                        self.active_queue.remove(current_model) # Banned for the rest of the run
                        
                elif "decommissioned" in error_msg:
                    print(f"    [STATE UPDATE] {current_model} is decommissioned. Permanently removing from router.")
                    if current_model in self.active_queue:
                        self.active_queue.remove(current_model) # Banned forever
                        
                else:
                    print(f"    [ERROR] Bad Request on {current_model}: {e}")
            
            # 3. UNEXPECTED ERRORS
            except Exception as e:
                print(f"    [ERROR] Unexpected failure on {current_model}: {type(e).__name__}")

            if attempt == len(execution_queue) - 1:
                print(f"    [FATAL] All fallback models exhausted.")
                return "SYSTEM_ERROR: API_EXHAUSTED"

        return "SYSTEM_ERROR: UNKNOWN"