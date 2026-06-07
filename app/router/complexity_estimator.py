class ComplexityEstimator:
    def estimate(self,query):
        score = 0

        reasoning_words = [
            "why","analyze","explain",
            "compare","contrast","evaluate",
            "architecture","optimize","design"]
        
        for word in reasoning_words:
            if word in query.lower():
                score += 0.1
        score += min(len(query) / 500, 0.5)

        return min(score,1.0)