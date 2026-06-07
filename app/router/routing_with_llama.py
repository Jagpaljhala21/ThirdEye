import os
import json
from dotenv import load_dotenv
from groq import Groq, RateLimitError  # Explicitly import RateLimitError

load_dotenv()

class RoutingWithLlama:

    def __init__(self, user_query):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        # Define both primary and fallback architectures
        self.primary_model = "llama-3.1-8b-instant"
        self.fallback_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        self.query = user_query
        self.system_query = """your are a cognitive router for thirdeye.analyze the user query.
        output a strict json object with exact these keys:
        - "complexity" (float 0.0 to 1.0)
        - "uncertainty" (float 0.0 to 1.0)
        - "module" (string: either "shallow_reasoning" or "deep_reasoning" based on the complexity and uncertainty)
        - "task" (string: like coding debug based on the user query)
        - "sub-task" (string the domain of the task its related like concurrency_control is coding like this )
        Do not give output in none or any other format just in integer or string or float as mentioned above. DO NOT give any explanation or text other than the json object.
        DO NOT OUTPUT ANY OTHER TEXT OR EXPLANATIION.
        """
    
    def _execute_completion(self, model_name):
        """Helper method to handle the standard completion call logic."""
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": self.system_query},
                {"role": "user", "content": self.query}
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=80
        )
        return json.loads(response.choices[0].message.content)

    def routing(self):
        try:
            # 1. Attempt generation with the ultra-fast primary model
            return self._execute_completion(self.primary_model)
            
        except RateLimitError:
            # 2. Intercept 429 errors and automatically scale to the fallback model
            print(f"[Warning] Rate limit hit for {self.primary_model}. Transitioning to {self.fallback_model}...")
            try:
                return self._execute_completion(self.fallback_model)
            except Exception as fallback_error:
                print(f"[Critical Error] Fallback model execution failed: {fallback_error}")
                raise fallback_error
                
        except Exception as general_error:
            # Catch other unexpected API exceptions (e.g., authentication or validation errors)
            print(f"[Error] Request failed due to unexpected API error: {general_error}")
            raise general_error
'''
query = """
Analyze this distributed system architecture
and explain why latency increases under heavy traffic.
""" 
test = RoutingWithLlama(user_query=query)
t = test.routing()

print(t['complexity'])
'''
