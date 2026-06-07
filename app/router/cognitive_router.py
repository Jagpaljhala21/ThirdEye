from app.router.routing_with_llama import RoutingWithLlama

class CognitiveRouter:
    
    def route(self, query):
        routing_obj = RoutingWithLlama(user_query=query)
        routing_ans = routing_obj.routing()
        
        complexity = routing_ans["complexity"]
        uncertainty = routing_ans["uncertainty"]
        module = routing_ans["module"]
        task = routing_ans["task"]
        
        # Initialize the downstream execution control flag
        bypass_graph = False
        
        # 1. Short-Circuit Pathway (Intercepts simple queries like "What is the capital of India")
        if complexity <= 0.3 and uncertainty <= 0.3:
            model = "llama-3.1-8b-instant"
            module = "direct_short_circuit"
            bypass_graph = True
            
        # 2. High Complexity & High Uncertainty -> Heavy Reasoning Engine
        elif complexity > 0.6 and uncertainty > 0.3:
            model = "llama-3.3-70b-versatile"
            
        # 3. Medium Complexity / High Uncertainty -> Balanced Scout Engine
        elif uncertainty > 0.4 and complexity <= 0.5:
            model = "meta-llama/llama-4-scout-17b-16e-instruct"
            
        # 4. Standard Fallback / Low Uncertainty Mixed Boundaries
        else:
            model = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        return {
            "task": task,
            "sub-task": routing_ans["sub-task"],
            "model": model,
            "complexity": complexity,
            "uncertainty": uncertainty,
            "modules": module,
            "bypass_graph": bypass_graph  # Downstream orchestrator handles this handshake
        }