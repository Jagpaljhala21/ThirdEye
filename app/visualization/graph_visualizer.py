import re
from pyvis.network import Network

class GraphVisualizer:

    def _md_to_html(self, text: str) -> str:
        """Convert markdown bold/italic to HTML."""
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        return text

    def _get_color(self, confidence: float) -> str:
        """Return color based on confidence threshold."""
        if confidence >= 0.78:
            return "#00ff88"   # green  — high confidence
        elif confidence >= 0.70:
            return "#ffb347"   # orange — medium confidence
        else:
            return "#ff4d4d"   # red    — low confidence

    def _get_edge_color(self, graph, source: str, target: str) -> str:
        """Color edge based on confidence drop between nodes."""
        src_conf = graph.nodes[source].get("confidence", 1.0)
        tgt_conf = graph.nodes[target].get("confidence", 1.0)
        drop = src_conf - tgt_conf

        if drop > 0.1:
            return "#ff4d4d"   # red   — big confidence drop
        elif drop > 0.0:
            return "#ffb347"   # orange — slight drop
        else:
            return "#00ff88"   # green  — stable or improving

    def visualize(
        self,
        graph,
        output_path: str = "reasoning_graph.html"
    ):
        net = Network(
            height="900px",
            width="100%",
            bgcolor="#0d1117",
            font_color="white",
            directed=True
        )

        # add nodes
        for node, data in graph.nodes(data=True):

            confidence = data.get("confidence", 1.0)
            output = data.get("output", "No output generated.")

            color = self._get_color(confidence)
            output_html = self._md_to_html(output)

            # html tooltip with rendered markdown (displays on hover)
            tooltip = f"""
            <div style="
                max-width:500px;
                padding:10px;
                background:#161b22;
                color:white;
                border-radius:10px;
                font-family:Arial;
            ">
                <h2 style="color:#58a6ff;">{node}</h2>
                <p><b>Confidence:</b> {confidence}</p>
                <hr>
                <p style="white-space:pre-wrap; line-height:1.5;">
                    {output_html}
                </p>
            </div>
            """

            # Clean the preview text to prevent renderer crashes from newlines/ticks
            clean_preview = re.sub(r'\s+', ' ', output).strip()
            
            # Create a clean, multi-line label
            label = f"{node}\n[{confidence}]\n\n{clean_preview[:80]}..."

            net.add_node(
                node,
                label=label,
                title=tooltip,
                color=color,
                shape="box",      # Uses a structured box instead of a dot
                borderWidth=2,
                font={"color": "#0d1117"} # Dark text inside the colored box for contrast
            )

        # add edges with confidence-drop coloring
        for source, target in graph.edges():
            edge_color = self._get_edge_color(graph, source, target)

            net.add_edge(
                source,
                target,
                color=edge_color,
                arrows="to",
                width=3
            )

        # --- THE FIX: Expanded Repulsion for Large Box Nodes ---
        net.set_options("""
        var options = {
          "nodes": {
            "font": {
              "size": 16,
              "face": "arial",
              "bold": true,
              "color": "#0d1117"
            },
            "margin": 15,
            "widthConstraint": {
              "maximum": 300 
            }
          },
          "layout": {
            "hierarchical": {
              "enabled": false
            }
          },
          "physics": {
            "enabled": true,
            "solver": "repulsion",
            "repulsion": {
              "nodeDistance": 450,
              "springLength": 400,
              "centralGravity": 0.05,
              "springConstant": 0.05
            }
          },
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
          },
          "edges": {
            "smooth": {
              "type": "cubicBezier",
              "forceDirection": "horizontal",
              "roundness": 0.4
            }
          }
        }
        """)

        net.show(output_path, notebook=False)
        print(f"\nGraph saved as {output_path}")