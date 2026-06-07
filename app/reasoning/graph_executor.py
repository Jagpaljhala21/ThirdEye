import re
from app.reasoning.node_executor import NodeExecutor
from app.reasoning.critic_engine import CriticEngine  
from app.memory.memory_manager import MemoryManager # <-- Integrated Vector Memory

class GraphExecutor:
    def __init__(self):
        self.node_executor = NodeExecutor()
        self.critic = CriticEngine() 
        self.memory = MemoryManager() # <-- Initialized FAISS + SentenceTransformers

    def _build_pruned_context(self, original_query: str, execution_trace: list, max_full_history: int = 2) -> str:
        """
        Dynamically rebuilds the context to prevent token overflow.
        Integrates long-term semantic memory and prunes older steps.
        """
        # Injected Strict System Persona
        context = (
            "SYSTEM PERSONA: You are an expert technical AI architecture tutor.\n"
            "RULE 1: If the user asks for a conceptual explanation, provide a clear, concise, and structured text explanation.\n"
            "RULE 2: If the task requires code, provide ONLY the final, most highly optimized version of the script. Do NOT provide iterative drafts or multiple versions.\n"
            "RULE 3: Do NOT recommend video tutorials.\n\n"
        )
        # Pull past semantic memory from FAISS (returns empty string if no relevant match)
        past_memory = self.memory.retrieve_context(original_query)
        context += past_memory
        
        context += f"ORIGINAL SYSTEM TASK:\n{original_query}\n\n"
        
        if not execution_trace:
            return context
            
        context += "--- PAST EXECUTION PROGRESS ---\n"
        
        recent_steps = execution_trace[-max_full_history:]
        older_steps = execution_trace[:-max_full_history]
        
        for step in older_steps:
            context += f"[Completed Successfully] {step['node']}\n"
            
        if older_steps:
            context += "\n--- IMMEDIATE CONTEXT ---\n"
            
        for step in recent_steps:
            context += f"===== {step['node']} =====\n{step['output']}\n\n"
            
        context += "Based on the progress above, execute the next step."
        return context

    def execute(self, graph, query, llm_engine, model):
        execution_trace = []
        
        initial_nodes = [n[0] for n in graph.nodes(data=True)]
        final_node_name = initial_nodes[-1] if initial_nodes else "Final Synthesis"
        
        skip_to_synthesis = False

        for base_node_name in initial_nodes:
            if base_node_name in ["Self Evaluation", "Revised Reasoning"]:
                continue
                
            if skip_to_synthesis and base_node_name != final_node_name:
                print(f"[PRUNED] Skipping {base_node_name} due to short-circuit.")
                continue

            confidence = graph.nodes[base_node_name].get("confidence", 1.0)
            
            if base_node_name == final_node_name:
                node_token_limit = 3000  # Bumped to 3000 to prevent Final Synthesis from truncating
            else:
                node_token_limit = 2500  

            max_retries = 0 if base_node_name == final_node_name else 1
            attempts = 0
            current_node_name = base_node_name
            
            current_prompt = self._build_pruned_context(query, execution_trace)
            node_passed = False

            while attempts <= max_retries and not node_passed:
                print(f"Executing node: {current_node_name} with confidence {confidence}")

                output = self.node_executor.execute_node(
                    node_name=current_node_name,
                    context=current_prompt,
                    llm_engine=llm_engine,
                    model=model,
                    max_tokens=node_token_limit
                )

                clean_output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
                if not clean_output:
                    clean_output = output

                graph.nodes[current_node_name]["output"] = clean_output

                impossibility_flags = [
                    "mathematically impossible", "premise is invalid", 
                    "cannot be expressed", "impossible to", "fundamentally flawed",
                    "violates fundamental"
                ]
                if any(flag in clean_output.lower() for flag in impossibility_flags):
                    print(f"\n[SHORT-CIRCUIT] '{current_node_name}' identified an impossible premise. Fast-forwarding to Synthesis...")
                    skip_to_synthesis = True
                    node_passed = True
                    confidence = 0.99
                    break  

                if max_retries > 0:
                    print(f"--> Triggering Critic for {current_node_name}...")
                    evaluation = self.critic.evaluate_node(query, clean_output)
                    
                    # Ensure confidence is a float
                    try:
                        confidence = float(evaluation.get("confidence_score", confidence))
                    except (ValueError, TypeError):
                        pass
                        
                    graph.nodes[current_node_name]["confidence"] = confidence

                    # --- THE "GOOD ENOUGH" EARLY EXIT OPTIMIZATION ---
                    critic_passed = evaluation.get("passed", True)
                    
                    if critic_passed:
                        print(f"[SUCCESS] {current_node_name} passed evaluation natively.")
                        node_passed = True
                    elif confidence >= 0.90:
                        print(f"[ACCEPTABLE] {current_node_name} failed strict critique but achieved {confidence} confidence. Bypassing revisions to save tokens.")
                        node_passed = True
                    # -------------------------------------------------
                    else:
                        print(f"[FAIL] {current_node_name} failed. Confidence: {confidence}. Critique: {evaluation.get('critique')}")
                        attempts += 1
                        
                        if attempts <= max_retries:
                            next_node_name = f"{base_node_name} (Revision {attempts})"
                            graph.add_node(next_node_name, confidence=1.0)
                            graph.add_edge(current_node_name, next_node_name) 
                            
                            current_prompt = f"""
                            ORIGINAL TASK: {query}
                            YOUR PREVIOUS FLAWED OUTPUT: {clean_output}
                            SYSTEM CRITIQUE: {evaluation.get('critique')}
                            REQUIRED FIX: {evaluation.get('suggested_fix')}
                            
                            Rewrite the output to fix the flaw.
                            """
                            current_node_name = next_node_name
                        else:
                            print(f"\n[CRITICAL] Max retries hit for {base_node_name}. Triggering Cloud Bailout...")
                            
                            bailout_prompt = f"""
                            You are the ultimate fallback expert. The previous reasoning model failed this task completely.
                            ORIGINAL TASK: {query}
                            FAILED ATTEMPT: {clean_output}
                            CRITIQUE TO FIX: {evaluation.get('critique')}
                            REQUIRED FIX: {evaluation.get('suggested_fix')}
                            
                            Bypass all previous errors. Write the final, perfectly corrected output. 
                            """
                            
                            bailout_output = self.node_executor.execute_node(
                                node_name=f"{base_node_name} Bailout",
                                context=bailout_prompt,
                                llm_engine=llm_engine,
                                model=model, 
                                max_tokens=node_token_limit
                            )
                            
                            bailout_clean = re.sub(r'<think>.*?</think>', '', bailout_output, flags=re.DOTALL).strip()
                            if not bailout_clean:
                                bailout_clean = bailout_output
                                
                            bailout_node_name = f"{base_node_name} (Cloud Bailout)"
                            graph.add_node(bailout_node_name, confidence=0.99)
                            graph.add_edge(current_node_name, bailout_node_name)
                            graph.nodes[bailout_node_name]["output"] = bailout_clean
                            
                            clean_output = bailout_clean
                            confidence = 0.99
                            
                            print(f"[RESCUED] Cloud Bailout resolved the logic for {base_node_name}.")
                            node_passed = True 
                else:
                    node_passed = True

            if node_passed:
                execution_trace.append({
                    "node": current_node_name,
                    "confidence": confidence,
                    "output": clean_output
                })

        # --- AUTO-INDEXING SYSTEM (Save successful runs to memory) ---
        if execution_trace and not skip_to_synthesis:
            final_synthesis = execution_trace[-1]["output"]
            
            # TRUNCATION GUARD: Prevent saving broken/cut-off code to FAISS memory
            if not final_synthesis.strip().endswith(("```", ".", "}", ">", "\n", ";")):
                print("\n[WARNING] Final output appears truncated. Bypassing memory storage to prevent corruption.")
            else:
                lessons = "Ensure strict compliance with explicit layout and structural logic rules."
                self.memory.store_memory(
                    original_query=query,
                    final_output=final_synthesis,
                    lessons_learned=lessons
                )

        return execution_trace