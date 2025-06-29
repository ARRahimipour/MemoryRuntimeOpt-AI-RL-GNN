# Simulated Dataset Loader - Mimicking Code Optimization Dataset (e.g., CodeNet, LLM fine-tuning datasets)

# Sample pairs of unoptimized and optimized code snippets
data_samples = [
    {
        "id": "code1",
        "unoptimized": "for i in range(len(arr)): print(arr[i])",
        "optimized": "for x in arr: print(x)"
    },
    {
        "id": "code2",
        "unoptimized": "if a == True: do_something()",
        "optimized": "if a: do_something()"
    },
    {
        "id": "code3",
        "unoptimized": "x = x + 0",
        "optimized": "# redundant operation removed"
    }
]

# Simulated loader function
def load_simulated_code_dataset():
    print("Loaded {} code samples".format(len(data_samples)))
    for sample in data_samples:
        print(f"ID: {sample['id']}")
        print("- Unoptimized:", sample['unoptimized'])
        print("- Optimized:", sample['optimized'])
        print("---")

# Run this script directly
if __name__ == "__main__":
    load_simulated_code_dataset()
