# 🧠 Autonomous Cognitive Agent V1

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

## 🏗️ Architecture

1. **User Input:** The user submits a complex prompt via the Streamlit UI.
2. **Cognitive Router:** Assesses the prompt and routes it to the appropriate reasoning module.
3. **Graph Builder:** Generates a topological DAG of sequential tasks (e.g., *Analysis -> Design -> Code -> Validation*).
4. **Graph Executor & Critic:** Executes each node. The Critic evaluates the output. If flawed, it branches into a `(Revision)` node. If successful, it advances.
5. **Memory Manager:** Synthesizes the final verified output and indexes it into the local FAISS vector database.

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
git clone [https://github.com/YOUR_USERNAME/third-eye-agent.git](https://github.com/YOUR_USERNAME/third-eye-agent.git)
cd third-eye-agent
