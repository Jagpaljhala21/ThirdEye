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

    def generate_nodes(self, user_query: str, routing_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Dynamically analyzes the user query and routing metadata to plan 
        the optimal Directed Acyclic Graph (DAG) execution sequence.
        """
        # Base fallback topology in case of parsing failures (matches the new JSON schema)
        fallback_plan = [
            {"id": "node_1", "task": "Perform foundational analysis and extract core requirements from the input.", "depends_on": []},
            {"id": "node_2", "task": "Execute primary logical reasoning or task decomposition.", "depends_on": ["node_1"]},
            {"id": "node_3", "task": "Synthesize the final comprehensive answer.", "depends_on": ["node_1", "node_2"]}
        ]
        
        system_prompt = (
            "You are the Core Orchestrator and Workflow Planner for an advanced agentic framework.\n"
            "Your job is to analyze a user query and its routing metadata, then design a customized "
            "execution graph (Directed Acyclic Graph) of specialized processing nodes.\n\n"
            "Guidelines for building the graph:\n"
            "1. Simple queries should only take 1-2 sequential nodes.\n"
            "2. Highly complex technical tasks should be broken down into specific sub-tasks. "
            "Nodes that do not rely on each other should have empty 'depends_on' arrays so they can run in parallel.\n"
            "3. If self-evaluation or code verification is necessary, inject validation nodes.\n"
            "4. The final node must ALWAYS be a synthesis step that depends on all major previous branches.\n\n"
            "You must return your response strictly as a JSON object matching this exact schema:\n"
            "{\n"
            '  "plan": [\n'
            '    {"id": "node_1", "task": "Retrieve foundational knowledge about X", "depends_on": []},\n'
            '    {"id": "node_2", "task": "Write the code for Y", "depends_on": ["node_1"]},\n'
            '    {"id": "node_3", "task": "Synthesize the final answer", "depends_on": ["node_1", "node_2"]}\n'
            "  ]\n"
            "}\n"
            "Ensure that every string in a 'depends_on' array exactly matches the 'id' of a preceding node."
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
                max_tokens=1024   # Increased slightly to allow for complex graph arrays
            )
            
            # Parse the structured JSON response
            response_data = json.loads(response.choices[0].message.content)
            plan = response_data.get("plan", fallback_plan)
            
            # Absolute safety guardrails: ensure plan is a non-empty list
            if not isinstance(plan, list) or len(plan) == 0:
                return fallback_plan
                
            return plan

        except Exception as e:
            # Graceful degradation to standard path if API fails or times out
            print(f"[WARNING] NodeGenerator failed with error: {e}. Falling back to default topology.")
            return fallback_plan
