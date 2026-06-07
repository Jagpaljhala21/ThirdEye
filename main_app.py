import streamlit as st
import streamlit.components.v1 as components

from app.router.cognitive_router import CognitiveRouter
from app.reasoning.graph_builder import ReasoningGraphBuilder
from app.reasoning.confidence_propagation import ConfidencePropagation
from app.reasoning.graph_executor import GraphExecutor
from app.models.llm_engine import LLMEngine
from app.visualization.interactive_visualizer import InteractiveVisualizer

st.set_page_config(page_title="Cognitive Agent V1", layout="wide", page_icon="🧠")

def load_custom_css():
    st.markdown("""
        <style>
        .stTextArea textarea {
            border-radius: 10px;
            border: 1px solid #333;
            background-color: #1E212B;
        }
        .stButton>button[kind="primary"] {
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 204, 150, 0.2);
        }
        .stButton>button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 204, 150, 0.4);
        }
        .stAlert {
            border-radius: 10px;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

st.title("🧠 ThirdEye Workspace")

# --- CACHING LAYER ---
@st.cache_resource
def get_cognitive_engines():
    print("[SERVER] Booting Cognitive Engines into Cache...")
    return {
        "router": CognitiveRouter(),
        "builder": ReasoningGraphBuilder(),
        "propagator": ConfidencePropagation(),
        "executor": GraphExecutor(), 
        "llm": LLMEngine()
    }

engines = get_cognitive_engines()

# --- USER INPUT ---
query = st.text_area("Enter your prompt:", height=100, placeholder="Ask a complex software engineering question...")

if st.button("Initialize Pipeline", type="primary"):
    if not query.strip():
        st.warning("Please enter a prompt.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            with st.expander("🛠️ View Agent Execution Trace", expanded=False):
                with st.status("Booting Cognitive Pipeline...", expanded=True) as status:
                    
                    st.write("📡 Routing query to optimal model...")
                    routing_info = engines["router"].route(query)
                    
                    st.write("🏗️ Synthesizing dynamic execution topology...")
                    graph = engines["builder"].build_graph(query=query, routing_info=routing_info)
                    graph = engines["propagator"].propagate(graph, routing_info["complexity"])
                    
                    st.write("⚙️ Executing nodes and triggering Critic...")
                    
                    with st.spinner("Agent is actively thinking and self-correcting..."):
                        trace = engines["executor"].execute(
                            graph=graph,
                            query=query,
                            llm_engine=engines["llm"],
                            model=routing_info["model"]
                        )
                    
                    status.update(label="Pipeline Execution Complete!", state="complete", expanded=False)

            st.subheader("Final Synthesis")
            if trace:
                with st.container(border=True):
                    st.markdown(trace[-1]["output"])

        with col2:
            st.subheader("Live Architecture Graph")
            visualizer = InteractiveVisualizer()
            html_graph = visualizer.generate_html(graph)
            components.html(html_graph, height=650)
