from app.models.llm_engine import LLMEngine
class NodeExecutor:

    def execute_node(
        self,
        node_name,
        context,
        llm_engine,
        model,
        max_tokens=None
    ):
        base_instruction = "CRITICAL: DO NOT use <think> tags. Do not write out your inner monologue. Respond immediately with the requested data.\n\n"
        prompts = {

            "Input Analysis": base_instruction + 
            f"""
            Analyze the user's query.

            Return ONLY:
            - core problem
            - important entities
            - main objective

            Keep response concise.

            Context:
            {context}
            """,

            "Task Decomposition": base_instruction +
            f"""
            Break the problem into reasoning steps.

            Return:
            - subproblems
            - dependencies
            - important bottlenecks

            Use short bullet points only.

            Context:
            {context}
            """,

            "Deep Reasoning": base_instruction +
            f"""
            Perform deep reasoning on the problem.

            Focus on:
            - causal relationships
            - bottlenecks
            - tradeoffs
            - system behavior

            Return concise but high-quality analysis.

            Avoid repetition.

            Keep under 500 words.

            Context:
            {context}
            """,

            "Self Evaluation": base_instruction +
            f"""
            Critically evaluate previous reasoning.

            Detect:
            - weak assumptions
            - missing evidence
            - contradictions
            - unclear conclusions

            Return concise critique only.

            Context:
            {context}
            """,

            "Final Synthesis": base_instruction +
            f"""
            Generate a final structured answer.

            DO NOT restate findings already covered in prior nodes.
            Only add: root cause summary, recommendations, action plan.

            Requirements:
            - concise
            - non-repetitive  
            - actionable

            Keep under 400 words.

            Context:
            {context}
            """
        }

        prompt = prompts.get(
            node_name,
            context
        )

        response = llm_engine.generate(
            model=model,
            prompt=prompt,
            max_completion_tokens=max_tokens
        )

        return response