from cognitive_router import CognitiveRouter

router = CognitiveRouter()
query = """
Anaylze this distributedd system architecture and explain how to optimize it for better performance
and why latency is high when processing large datasets
"""

result = router.route(query)

print(result)