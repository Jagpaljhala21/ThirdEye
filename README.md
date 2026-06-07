# 🧠 ThirdEYE V1

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg)
![NetworkX](https://img.shields.io/badge/NetworkX-DAGs-lightgrey.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-00CC96.svg)

An advanced, self-correcting autonomous AI agent built with a Directed Acyclic Graph (DAG) reasoning architecture. It features dynamic routing, a rigorous self-evaluating Critic loop, semantic long-term memory, and a live interactive web UI.

---

## ✨ Key Features

* **Dynamic Cognitive Routing:** Analyzes incoming queries to determine complexity, uncertainty, and the optimal LLM to handle the task.
* **Self-Healing DAG Planner:** Maps out a multi-step execution topology. If a step fails, the integrated **Critic Engine** automatically triggers a targeted revision loop until a confidence threshold (e.g., >90%) is met.
* **Semantic Long-Term Memory:** Uses `sentence-transformers` and `FAISS` to semantically index successful pipeline executions, allowing the agent to "remember" past solutions and avoid repeating mistakes.
* **Live Interactive Visualizer:** Renders the agent's internal thought process and topological execution path in real-time using `PyVis`. Revision nodes are visually clustered with dotted paths and color-coded by confidence scores.
* **Safe & Optimized Execution:** Includes token truncation guards, infinite-loop breakers, and aggressive constraint prompting to ensure fast, lean inference.

---
```mermaid
graph TD
    %% Styling Definitions
    classDef user fill:#0E1117,stroke:#00CC96,stroke-width:2px,color:#FFF;
    classDef core fill:#1E212B,stroke:#555,stroke-width:1px,color:#FFF;
    classDef critic fill:#3B1C1C,stroke:#FF4B4B,stroke-width:2px,color:#FFF;
    classDef memory fill:#112B22,stroke:#00CC96,stroke-width:2px,color:#FFF;
    classDef output fill:#1E212B,stroke:#FFA421,stroke-width:2px,color:#FFF;

    %% Nodes
    User(("🧑‍💻 User Input")):::user
    UI["💻 Streamlit UI"]:::core
    
    Router{"📡 Cognitive Router"}:::core
    Builder["🏗️ Graph Builder (NetworkX)"]:::core
    Propagator["📊 Confidence Propagation"]:::core
    
    Executor(("⚙️ Graph Executor")):::core
    Critic{"⚖️ Critic Engine"}:::critic
    
    Memory["🧠 Memory Manager"]:::memory
    FAISS[("🗄️ FAISS Vector DB")]:::memory
    
    Visualizer["🌐 Interactive Visualizer"]:::output
    Synthesis["📝 Final Synthesis"]:::output

    %% Flow
    User -->|Prompt| UI
    UI --> Router
    Router -->|Routing Info| Builder
    Builder -->|Initial DAG| Propagator
    Propagator -->|Scored DAG| Executor
    
    %% Execution & Critic Loop
    Executor -->|Node Output| Critic
    Critic -.->|Fail: Trigger Revision Loop| Executor
    Critic ==>|Pass: Confidence > Threshold| Memory
    
    %% Memory Interactions
    Memory <-->|Semantic Search & Store| FAISS
    Memory -->|Validated Output| Synthesis
    
    %% UI Rendering
    Executor -->|Execution Trace| Visualizer
    Visualizer -->|PyVis HTML| UI
    Synthesis -->|Markdown| UI

```
---

## 💻 Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (with custom CSS injection & dark mode)
* **Graph Visualization:** [PyVis](https://pyvis.network/) (Hierarchical HTML physics engine)
* **Graph Mathematics:** [NetworkX](https://networkx.org/)
* **Vector Memory:** [FAISS](https://github.com/facebookresearch/faiss) (CPU) + HuggingFace `all-MiniLM-L6-v2`
* **LLM Engine:** Groq / OpenAI / Anthropic (Configurable via Environment Variables)

---

## 🚀 Local Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Jagpaljhala21/ThirdEye.git
cd ThirdEye
