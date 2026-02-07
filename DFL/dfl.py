import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)  # Changed num_classes to 10 for MNIST

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# Load the MNIST dataset
trans_mnist = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
test_dataset = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

# Define decentralized federated training parameters
num_clients = 5
batch_size = 64
num_epochs = 100  # Reduced the number of epochs for this example

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001

# Create a list to store client models
client_models = []
client_statistics = []

# Define a function for training a model on a client's data
def train_client_model(client_model, client_train_loader):
    optimizer = optim.SGD(client_model.parameters(), lr=learning_rate, momentum=0.9)
    total_step = len(client_train_loader)

    client_loss = []
    client_accuracy = []
    client_time = []

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for i, (images, labels) in enumerate(client_train_loader):
            # Forward pass
            outputs = client_model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()

        # Calculate average loss for the epoch
        epoch_loss /= total_step
        epoch_accuracy = 100.0 * epoch_correct / epoch_total

        # Calculate the time
        end_time = time.time()
        epoch_time = end_time - start_time

        # Print client's epoch results
        print(
            f"Client - Epoch [{epoch + 1}/{num_epochs}], Time: {epoch_time:.2f}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Append the loss and accuracy to the client's statistics
        client_loss.append(epoch_loss)
        client_accuracy.append(epoch_accuracy)
        client_time.append(epoch_time)

        # Simulate DFL: Share model updates with other clients periodically (e.g., every 10 epochs)
        # if (epoch + 1) % 10 == 0:
            # Share the client's model with other clients (simplified DFL)
            # share_model_with_other_clients(client_model)
        share_model_with_other_clients(client_model)

    return client_loss, client_accuracy, client_time

# Simulated function for sharing a client's model with other clients (simplified DFL)
def share_model_with_other_clients(client_model):
    for other_client_model in client_models:
        other_client_model.load_state_dict(client_model.state_dict())

# Split the dataset into clients
client_data_size = len(train_dataset) // num_clients
client_datasets = [torch.utils.data.Subset(train_dataset, range(i * client_data_size, (i + 1) * client_data_size)) for i
                   in range(num_clients)]

# Train client models
for client_id, client_dataset in enumerate(client_datasets):
    client_model = CNN()
    client_train_loader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
    client_loss, client_accuracy, client_time = train_client_model(client_model, client_train_loader)
    client_models.append(client_model)
    client_statistics.append((client_loss, client_accuracy, client_time))

# Calculate and print the average loss and accuracy across all clients for each epoch
for epoch in range(num_epochs):
    avg_loss = sum([client_stats[0][epoch] for client_stats in client_statistics]) / num_clients
    avg_accuracy = sum([client_stats[1][epoch] for client_stats in client_statistics]) / num_clients

    # Calculate the total time for the epoch
    epoch_times = [client_stats[2][epoch] for client_stats in client_statistics]
    total_epoch_time = sum(epoch_times)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.2f}%, Total Time: {total_epoch_time:.2f} seconds")

# Now you can use any of the client_models for inference or further training.
print("Decentralized Federated Learning completed!")
