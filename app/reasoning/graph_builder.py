import networkx as nx

from app.reasoning.node_generator import NodeGenerator

class ReasoningGraphBuilder:
    def __init__(self):
        self.node_generator = NodeGenerator()
    def build_graph(self,query,routing_info):
        graph = nx.DiGraph()
        nodes = self.node_generator.generate_nodes(user_query=query, routing_info=routing_info)
        #add nodes
        for node in nodes:
            graph.add_node(
                node,
                confidence = 1.0
                )
        #sequential edges
        for i in range(len(nodes)-1):
            graph.add_edge(nodes[i],nodes[i+1])
        return graph