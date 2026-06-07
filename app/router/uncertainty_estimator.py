class UncertaintyEstimator:
    def estimate(self,query):
        uncertainty = 0
        ambiguous_words = [
            "maybe","perhaps","possibly",
            "uncertain","doubt","guess",
            "might","could","seems"]
        
        for word in ambiguous_words:
            if word in query.lower():
                uncertainty += 0.1

        return min(uncertainty,1.0)