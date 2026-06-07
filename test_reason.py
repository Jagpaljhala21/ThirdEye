from app.router.cognitive_router import CognitiveRouter
from app.visualization.graph_visualizer import (
    GraphVisualizer
)
from app.reasoning.graph_builder import (
    ReasoningGraphBuilder
)

from app.reasoning.confidence_propagation import (
    ConfidencePropagation
)

from app.reasoning.graph_executor import (
    GraphExecutor
)

from app.models.llm_engine import (
    LLMEngine
)

from app.models.bypass_logic import (
    BypassLogic
)

query = """
explain RAG and Agents
"""

# ROUTER
router = CognitiveRouter()

routing_info = router.route(query)


print("\n[ROUTER OUTPUT]")
print(routing_info)

if routing_info["bypass_graph"]:
    bypass_logic = BypassLogic(query=query)
    response = bypass_logic.bypass()
    print("\n[BYPASS RESPONSE]")
    print(response)
    exit(0)


# GRAPH
builder = ReasoningGraphBuilder()

graph = builder.build_graph(query=query, routing_info=routing_info)

# CONFIDENCE
propagator = ConfidencePropagation()

graph = propagator.propagate(
    graph,
    routing_info["complexity"]
)

# LLM
llm = LLMEngine()

# EXECUTION
executor = GraphExecutor()

trace = executor.execute(
    graph=graph,
    query=query,
    llm_engine=llm,
    model=routing_info["model"]
)

visualizer = GraphVisualizer()

visualizer.visualize(graph)

print("\n[EXECUTION TRACE]\n")

for step in trace:

    print("=" * 50)

    print(f"NODE: {step['node']}")

    print(f"CONFIDENCE: {step['confidence']}")

    print(f"\nOUTPUT:\n{step['output']}")