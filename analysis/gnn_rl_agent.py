import json
import random

# Simulated graph as input (usually comes from analysis/graph_analysis.py)
def load_graph(file_path="results/cfg_graph.json"):
    with open(file_path, "r") as f:
        graph = json.load(f)
    return graph

# Simplified RL agent: Simulate action selection based on node features (such as high runtime or memory usage)
def suggest_optimizations(graph):
    actions = []
    for node in graph.get("nodes", []):
        if node.get("runtime", 0) > 2.0:  # High runtime
            actions.append({
                "target": node["name"],
                "action": "move_to_gpu"
            })
        if node.get("memory", 0) > 100:  # High memory consumption
            actions.append({
                "target": node["name"],
                "action": "clear_cache"
            })
    return actions

if __name__ == "__main__":
    g = load_graph()
    suggestions = suggest_optimizations(g)
    print("[RL+GNN Agent] Suggested actions:")
    for s in suggestions:
        print(f" - {s['action']} --> {s['target']}")
