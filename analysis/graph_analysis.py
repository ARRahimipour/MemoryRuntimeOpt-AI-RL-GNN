import ast
import networkx as nx
import matplotlib.pyplot as plt
import time
import datetime
import os
import json
import random

# Read Python code for AST analysis
with open("profiling/baseline_training.py", "r") as f:
    code = f.read()
parsed = ast.parse(code)
cfg = nx.DiGraph()

# Class to build the graph
class CFGBuilder(ast.NodeVisitor):
    def __init__(self):
        self.prev_node = None

    def visit_FunctionDef(self, node):
        cfg.add_node(node.name, type="function", runtime=round(random.uniform(1.0, 4.0), 2), memory=random.randint(50, 200))
        if self.prev_node:
            cfg.add_edge(self.prev_node, node.name)
        self.prev_node = node.name
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            callee = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee = node.func.attr
        else:
            callee = "unknown"
        cfg.add_node(callee, type="call", runtime=round(random.uniform(0.5, 2.5), 2), memory=random.randint(30, 100))
        if self.prev_node:
            cfg.add_edge(self.prev_node, callee)
        self.prev_node = callee
        self.generic_visit(node)

CFGBuilder().visit(parsed)

# Save the graph as an image
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("results", exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.spring_layout(cfg, seed=42)
nx.draw(cfg, pos, with_labels=True, node_size=2500, node_color="skyblue",
        font_size=8, font_weight='bold', arrows=True, ax=ax)
plt.title("CFG (Control Flow Graph)")
plt.savefig(f"results/cfg_graph_{timestamp}.png", bbox_inches='tight')
plt.close()

# Save in JSON format for GNN/RL
json_data = {
    "nodes": [
        {"name": n, **cfg.nodes[n]} for n in cfg.nodes
    ],
    "edges": [
        {"source": u, "target": v} for u, v in cfg.edges
    ]
}
with open("results/cfg_graph.json", "w") as f:
    json.dump(json_data, f, indent=2)

print("[CFG Analysis] Graph extracted and saved as JSON and PNG.")
