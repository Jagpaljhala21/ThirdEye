import networkx as nx
from app.reasoning.node_generator import NodeGenerator

class ReasoningGraphBuilder:
    def __init__(self):
        self.node_generator = NodeGenerator()

    def build_graph(self, query, routing_info):
        """
        Builds a Directed Acyclic Graph (DAG) based on the LLM's dynamic execution plan.
        """
        # 1. Ask the Planner to generate the execution blueprint
        # Expected format from NodeGenerator: 
        # [{"id": "node_1", "task": "...", "depends_on": []}, ...]
        plan = self.node_generator.generate_nodes(user_query=query, routing_info=routing_info)
        
        # Initialize the Directed Graph
        graph = nx.DiGraph()
        
        # 2. Add all nodes and store the task instructions inside them
        for step in plan:
            graph.add_node(
                step["id"], 
                task=step["task"],
                confidence=1.0  # Initializing your confidence tracker
            )
            
        # 3. Draw the dependency edges (the arrows)
        for step in plan:
            for dep in step.get("depends_on", []):
                # Ensure the dependency actually exists before drawing the line
                if graph.has_node(dep):
                    graph.add_edge(dep, step["id"])
                    
        # 4. Critical Safety Check: Ensure no infinite circular loops
        if not nx.is_directed_acyclic_graph(graph):
            # If the LLM hallucinates a loop (e.g., A depends on B, B depends on A), 
            # we catch it here before it crashes the execution engine.
            raise ValueError("[CRITICAL] LLM generated a circular dependency. Triggering bailout...")
            
        return graph
