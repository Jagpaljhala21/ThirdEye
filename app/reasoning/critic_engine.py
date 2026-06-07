import json
import os
from groq import Groq

class CriticEngine:
    def __init__(self):
        # Using the fast API strictly for grading
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.evaluator_model = "openai/gpt-oss-120b"

    def evaluate_node(self, task_prompt: str, node_output: str) -> dict:
        """
        Grades the output and returns a strict JSON dictionary.
        """
        system_prompt = """
        CRITICAL PARADOX RULE: 
        If the user's original request is mathematically, logically, or physically impossible, the execution node MUST explicitly refute the premise. If the node correctly identifies and explains why the request is impossible, you must evaluate it as a [SUCCESS], even if it fails to fulfill the impossible constraints of the original prompt.
        
        You are the ThirdEYE System Critic. Evaluate the execution output against the requested task.
        Look for logic flaws, resource bottlenecks, or hallucinations.
        
        Respond ONLY in valid JSON format:
        {
            "passed": boolean,
            "confidence_score": float (0.0 to 1.0),
            "critique": "string explaining the exact flaw, or 'None'",
            "suggested_fix": "string explaining how to correct it, or 'None'"
        }
        """
        
        user_prompt = f"TASK:\n{task_prompt}\n\nOUTPUT TO EVALUATE:\n{node_output}"
        
        response = self.client.chat.completions.create(
            model=self.evaluator_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        return json.loads(response.choices[0].message.content)