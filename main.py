import subprocess
import os
import sys

scripts = [
    ("Loading Simulated Dataset", "dataset/simulate_dataset_loader.py"),
    ("Running Baseline Training", "profiling/baseline_training.py"),
    ("Analyzing Control Flow Graph (CFG)", "analysis/graph_analysis.py"),
    ("Running RL+GNN Agent", "analysis/gnn_rl_agent.py"),
    ("Running Optimized Training", "profiling/optimized_training.py"),
    ("Generating Resource Usage Graph", "resource_usage_graph.py")
]

python_exec = sys.executable  # Use current Python interpreter
print("\n==============================")
print(" MEMORYRUNTIMEOPT-AI FULL DEMO")
print("==============================\n")

for label, path in scripts:
    print(f"[+] {label}...")
    result = subprocess.run([python_exec, path], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("[ERROR]", result.stderr.strip())
    print("\n------------------------------\n")

print("✅ Done. Check the 'results/' folder for graphs, JSON, and usage reports.")