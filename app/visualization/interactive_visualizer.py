from pyvis.network import Network
import networkx as nx

class InteractiveVisualizer:
    def generate_html(self, graph: nx.DiGraph) -> str:
        """
        Converts the NetworkX DAG into a live, hierarchical PyVis network.
        Groups revision nodes visually with their base nodes.
        """
        net = Network(
            height="600px", 
            width="100%", 
            directed=True, 
            bgcolor="#0E1117", 
            font_color="white",
            cdn_resources='remote' 
        )
        
        # Hierarchical layout for a clean, top-down pipeline flow
        net.options.layout = {
            "hierarchical": {
                "enabled": True,
                "direction": "UD", 
                "sortMethod": "directed",
                "nodeSpacing": 150,
                "levelSeparation": 150
            }
        }
        
        net.toggle_physics(False)

        for node, data in graph.nodes(data=True):
            confidence = float(data.get("confidence", 1.0))
            is_revision = "(Revision" in node or "(Cloud Bailout)" in node
            
            # Color-code based on the Critic's confidence score
            if confidence >= 0.90:
                color = "#00CC96"  # Green
            elif confidence >= 0.70:
                color = "#FFA421"  # Orange
            else:
                color = "#FF4B4B"  # Red
                
            shape = "ellipse" if is_revision else "box"
            size = 15 if is_revision else 25

            net.add_node(
                n_id=node, 
                label=node, 
                title=f"Confidence: {confidence}", 
                color=color,
                shape=shape,
                size=size
            )

        for source, target in graph.edges():
            is_revision_edge = "(Revision" in target or "(Cloud Bailout)" in target
            
            net.add_edge(
                source, 
                target, 
                color="#888888",
                dashes=is_revision_edge, # Dotted line for revisions
                width=2 if not is_revision_edge else 1
            )

        return net.generate_html()