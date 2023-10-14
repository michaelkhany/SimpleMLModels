import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# Load data
file_path = "World Energy Overview.json"

with open(file_path, 'r') as file:
    data = json.load(file)

# Preprocess data
features = np.array([[float(x["Total Fossil Fuels Production"]), 
                      float(x["Nuclear Electric Power Production"]), 
                      float(x["Total Renewable Energy Production"]), 
                      float(x["Primary Energy Imports"]), 
                      float(x["Primary Energy Exports"]), 
                      float(x["Primary Energy Net Imports"]), 
                      float(x["Primary Energy Stock Change and Other"]), 
                      float(x["Total Fossil Fuels Consumption"]), 
                      float(x["Nuclear Electric Power Consumption"]), 
                      float(x["Total Renewable Energy Consumption"])] for x in data])
labels = np.array([float(x["Total Primary Energy Consumption"]) for x in data])

# Split the data into training and testing sets
train_features = torch.tensor(features[:-10], dtype=torch.float32)
test_features = torch.tensor(features[-10:], dtype=torch.float32)
train_labels = torch.tensor(labels[:-10], dtype=torch.float32).view(-1, 1)
test_labels = torch.tensor(labels[-10:], dtype=torch.float32).view(-1, 1)

# Create DataLoader for training and testing data
train_data = TensorDataset(train_features, train_labels)
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

# Define a neural network model for makespan prediction
class MakespanNN(nn.Module):
    def __init__(self):
        super(MakespanNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, define the loss function and the optimizer
model = MakespanNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the model
model_save_path = "MakespanModel/makespan_nn_model.pth"

# Check if the model file already exists
if not os.path.isfile(model_save_path):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
else:
    print("Model file already exists. Model was not saved to avoid overwriting.")

# Load the model
model_load_path = "MakespanModel/makespan_nn_model.pth"
loaded_model = MakespanNN()
loaded_model.load_state_dict(torch.load(model_load_path))

# Evaluate the model
loaded_model.eval()
with torch.no_grad():
    test_predictions = loaded_model(test_features)
    loss = criterion(test_predictions, test_labels)
    print(f'Test Loss: {loss.item()}')

