from app.router.cognitive_router import CognitiveRouter
from app.reasoning.confidence_propagation import ConfidencePropagation
from app.reasoning.graph_executor import GraphExecutor
from app.reasoning.graph_builder import ReasoningGraphBuilder

query = """
Anaylze why distributed system become unstable under heavy traffic.
"""

router = CognitiveRouter()

routing_info = router.route(query)
print("\n[Router Output]")
print(routing_info)

builder = ReasoningGraphBuilder()

graph = builder.build_graph(query=query, routing_info=routing_info)

#confidence

propagator = ConfidencePropagation()

graph = propagator.propagate(graph, routing_info['complexity'])
#execution
executor = GraphExecutor()

trace = executor.execute(graph)
print("\n[Execution Trace]")
for step in trace:
    print(step)