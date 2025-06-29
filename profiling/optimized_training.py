import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import tracemalloc
import gc

# Optimized version of the model with memory improvements
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

# Dataset creation and normalization similar to the baseline
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.FakeData(size=5000, image_size=(1, 28, 28), transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Start profiling
total_time = 0
tracemalloc.start()
start_time = time.time()

model.train()
for epoch in range(3):
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            x = x.clone()

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        # Actions suggested by the RL agent
        del x, y, out, loss
        torch.cuda.empty_cache()
        gc.collect()

end_time = time.time()
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"[Optimized] Time: {end_time - start_time:.2f}s | Peak Memory: {peak / 1024 / 1024:.2f}MB")
