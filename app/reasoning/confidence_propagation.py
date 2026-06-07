import networkx as nx
class ConfidencePropagation:

    def propagate(self, graph, complexity):
        base = round(1.0 - (complexity * 0.15), 2)

        try:
            ordered_nodes = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            ordered_nodes = list(graph.nodes)
        
        for node in ordered_nodes:
            parents = list(graph.predecessors(node))

            if not parents:
                inherited_confidence = 1
            else:
                inherited_confidence = min(
                    graph.nodes[p].get('confidence',1) for p in parents
                )
            final_confidence = round(inherited_confidence*base,2)
            graph.nodes[node]['confidence'] = max(0.0,min(final_confidence,1))
        return graph