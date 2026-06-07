from app.models.llm_engine import LLMEngine

class BypassLogic:
    def __init__(self,query):
        self.llm_engine = LLMEngine()
        self.model = "llama-3.1-8b-instant"
        self.query = query
    def bypass(self):
        response = self.llm_engine.generate(
            model=self.model,
            prompt=self.query
        )
        return response