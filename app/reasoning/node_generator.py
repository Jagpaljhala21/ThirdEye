import os
import json
from typing import List, Dict, Any
from groq import Groq

class NodeGenerator:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """Initializes the Groq client and default planning model."""
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model = model

    def generate_nodes(self, user_query: str, routing_info: Dict[str, Any]) -> List[str]:
        """
        Dynamically analyzes the user query and routing metadata to plan 
        the optimal sequence of execution nodes via LLM.
        """
        # Base fallback topology in case of parsing failures
        fallback_nodes = ["Input Analysis", "Task Decomposition", "Final Synthesis"]
        
        system_prompt = (
            "You are the Core Orchestrator and Workflow Planner for an advanced agentic framework.\n"
            "Your job is to analyze a user query and its routing metadata, then design a customized, "
            "linear execution sequence of specialized processing nodes. Do not use generic steps.\n\n"
            "Guidelines for selecting nodes:\n"
            "1. Simple queries (e.g., basic definitions, quick lookups) should only take 1-2 nodes (e.g., ['Direct Answer']).\n"
            "2. Highly complex technical or logical tasks should include distinct domain-specific steps.\n"
            "3. If self-evaluation or code verification is necessary, inject validation nodes.\n"
            "4. Always ensure the workflow starts with a foundational analysis step and ends with a synthesis step.\n\n"
            "You must return your response strictly as a JSON object matching this schema:\n"
            "{\n"
            "  \"reasoning_path\": [\"Node Name 1\", \"Node Name 2\", \"Node Name 3\"]\n"
            "}"
        )

        user_content = (
            f"User Query: {user_query}\n"
            f"Routing Metadata: {json.dumps(routing_info)}"
        )

        try:
            # Call Groq API using JSON mode to guarantee structured output
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for highly deterministic execution planning
                max_tokens=500
            )
            
            # Parse the structured JSON response
            response_data = json.loads(response.choices[0].message.content)
            nodes = response_data.get("reasoning_path", fallback_nodes)
            
            # Absolute safety guardrails: ensure nodes is a non-empty list
            if not isinstance(nodes, list) or len(nodes) == 0:
                return fallback_nodes
                
            return nodes

        except Exception as e:
            # Graceful degradation to standard path if API fails or times out
            print(f"[WARNING] NodeGenerator failed with error: {e}. Falling back to default topology.")
            return fallback_nodes

