import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# Load the MNIST dataset
trans_mnist = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
test_dataset = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

# Define decentralized federated training parameters
num_clients = 7
batch_size = 64
num_epochs = 10  # You can adjust the number of epochs per client training

# Define the loss function and optimizer (not used here, but typically needed)
criterion = nn.CrossEntropyLoss()

# Create a list to store client models
client_models = [CNN() for _ in range(num_clients)]

def train_client_model(client_model, client_train_loader, epochs):
    optimizer = optim.SGD(client_model.parameters(), lr=0.001, momentum=0.9)
    total_step = len(client_train_loader)
    client_model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(client_train_loader):
            # Forward pass
            outputs = client_model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print loss and accuracy for each epoch
        epoch_loss /= total_step
        epoch_accuracy = 100 * correct / total
        #print(
        #    f"Client {client_id} - Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return client_model

# Define a function to evaluate model on test dataset
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    return 100 * correct / len(test_loader.dataset)


# Split the dataset into clients
client_data_size = len(train_dataset) // num_clients
client_datasets = [torch.utils.data.Subset(train_dataset, range(i * client_data_size, (i + 1) * client_data_size)) for i
                   in range(num_clients)]
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Define global model - this could be a fresh instance or a pre-trained model
global_model = CNN()

# Training loop
num_rounds = 50
for round in range(num_rounds):
    local_weights = []

    print(f"\n--- Round {round + 1} of {num_rounds} ---")

    # Distribute the model to each client and collect updates
    for client_id, client_dataset in enumerate(client_datasets):
        local_model = CNN()
        local_model.load_state_dict(global_model.state_dict())  # Start with the global model

        # Create data loader for client's data
        client_train_loader = torch.utils.data.DataLoader(client_dataset, batch_size=64, shuffle=True)

        # Train the local model
        trained_local_model = train_client_model(local_model, client_train_loader, epochs=5)

        # Save the local model weights
        local_weights.append(trained_local_model.state_dict())

    # Update global model: Aggregate local model weights
    global_state_dict = global_model.state_dict()
    for k in global_state_dict.keys():
        global_state_dict[k] = torch.mean(torch.stack([local_weights[i][k].float() for i in range(len(local_weights))]),
                                          0)
    global_model.load_state_dict(global_state_dict)

    # Evaluate global model
    global_accuracy = evaluate(global_model, torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True))
    print(f"Round {round + 1}: Global Model Accuracy: {global_accuracy}%")

print("Basic Federated Learning process completed!")
