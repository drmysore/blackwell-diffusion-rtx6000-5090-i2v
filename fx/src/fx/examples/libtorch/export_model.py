import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# Create and script the model
model = SimpleModel()
model.eval()

# Create example input for tracing
example_input = torch.randn(1, 10)

# Script the model
scripted_model = torch.jit.script(model)

# Save it
scripted_model.save("/tmp/model.pt")
print("Model saved to /tmp/model.pt")

# Test it in Python first
with torch.no_grad():
    output = scripted_model(example_input)
    print(f"Python output: {output.item()}")
