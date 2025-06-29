# MemoryRuntimeOpt-AI

## Introduction

`MemoryRuntimeOpt-AI` is a Python-based research project for **AI-driven runtime and memory usage optimization** in software systems. It simulates optimization workflows using machine learning models, analyzes control flow graphs (CFGs), and applies reinforcement learning (RL) combined with graph neural networks (GNNs) to suggest performance improvements.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/MemoryRuntimeOpt-AI.git
cd MemoryRuntimeOpt-AI
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate       # On Windows
pip install -r requirements.txt
```

## Requirements

- Python 3.10 or later
- Dependencies listed in `requirements.txt`

## Project Structure

```plaintext
MemoryRuntimeOpt-AI/
├── main.py                     # Runs the full scenario
├── requirements.txt
├── dataset/
│   └── simulate_dataset_loader.py
├── profiling/
│   ├── baseline_training.py
│   └── optimized_training.py
├── analysis/
│   ├── graph_analysis.py       # Extracts CFG + exports JSON
│   └── gnn_rl_agent.py         # Suggests optimizations (RL + GNN simulation)
├── results/
│   ├── cfg_graph.json
│   ├── cfg_graph_*.png
│   ├── resource_usage_*.png
│   └── other logs/graphs...
```

## Scenario and Workflow

### 1. Simulating Dataset Load

Provides toy examples of unoptimized/optimized code for analysis:

```
for i in range(len(arr)) → for x in arr
if a == True → if a
x = x + 0 → removed
```

### 2. Running Baseline Training

Trains a simple MLP on synthetic data. Logs execution time and peak memory.

**Sample:**
```
[Baseline] Time: 13.75s | Peak Memory: 0.55MB
```

### 3. Analyzing Control Flow Graph (CFG)

Parses Python AST to build a weighted CFG. Saves PNG graph + JSON structure:

```
[CFG Analysis] Graph extracted and saved as JSON and PNG.
```

### 4. RL + GNN Agent Suggestion

Loads the graph JSON and identifies performance bottlenecks. Suggests actions like:

```
move_to_gpu --> forward
clear_cache --> forward
```

### 5. Running Optimized Training

Applies suggestions: memory cleanup, GPU offloading, caching...

**Sample:**
```
[Optimized] Time: 37.68s | Peak Memory: 0.54MB
```

### 6. Resource Usage Visualization

Boxplots comparing CPU and memory usage:

- `resource_usage_*.png`

## Sample Output Files

- `cfg_graph_*.png` – control flow graph
- `cfg_graph.json` – GNN-compatible graph structure
- `resource_usage_*.png` – CPU/memory boxplots
- Console logs from each module

## Notes

This is a research prototype demonstrating **automated optimization** using static/dynamic profiling + AI-based agents. Accuracy of actions can be tuned/improved in future iterations using actual RL training.

---

> Developed as part of a Master’s thesis in Software Engineering titled:
> "**AI for Runtime and Memory Usage Optimization in Software**"
