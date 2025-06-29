import matplotlib.pyplot as plt
import numpy as np
import datetime

# Data for baseline and optimized
baseline_times = [1.79, 2.40, 2.15]  # Example execution times for baseline (seconds)
optimized_times = [3.85, 3.60, 3.45]  # Example execution times for optimized (seconds)
baseline_memory = [0.65, 0.65, 0.60]  # Example memory usage for baseline (MB)
optimized_memory = [0.64, 0.63, 0.62]  # Example memory usage for optimized (MB)

# Create subplots for CPU and Memory usage
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# CPU Usage Distribution
ax1.boxplot([baseline_times, optimized_times], labels=['Baseline', 'Optimized'])
ax1.set_title('CPU Usage Distribution')
ax1.set_ylabel('Execution Time (seconds)')

# Memory Usage Distribution
ax2.boxplot([baseline_memory, optimized_memory], labels=['Baseline', 'Optimized'])
ax2.set_title('Memory Usage Distribution')
ax2.set_ylabel('Memory Usage (MB)')

# Save the plots
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.tight_layout()
plt.savefig(f"results/resource_usage_{timestamp}.png")
plt.show()
