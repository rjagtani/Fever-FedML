from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split
import flwr as fl
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
from collections import OrderedDict

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data():
    """Load CIFAR-10 (training and test set)."""
    data = pd.read_csv("pytorch_fever/fever_encoded.csv")
    X = data.drop('quantity(kWh)', axis=1)
    # selected_features = ['manufacturer', 'year', 'version', 'power(kW)', 'odometer',
    #              'trip_distance(km)', 'tire_type', 'city',
    #             'driving_style', 'park_heating', 'month', 'weekend']
    selected_features = ['manufacturer', 'year', 'power(kW)', 'trip_distance(km)', 'tire_type', 'month']
    X = X[selected_features]
    y = data['quantity(kWh)']
    feature_names = list(X.columns)
    #print(feature_names)
    X = X.values
    y = y.values
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=42)
    trainset = TensorDataset(X_train,y_train)
    testset = TensorDataset(X_test, y_test)
    # transform = transforms.Compose(
    # [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    # trainset = CIFAR10(".", train=True, download=True, transform=transform)
    # testset = CIFAR10(".", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples, np.array(X_val), np.array(y_val), feature_names

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for features, targets in trainloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(features).reshape(-1,1), targets.reshape(-1,1))
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.MSELoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            features, targets = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(features)
            loss = criterion(outputs.reshape(-1,1), targets.reshape(-1,1))
    accuracy = r2_score(targets, outputs)
    return loss, accuracy

class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Load model and data
net = Net(6, 64, 32, 1).to(DEVICE)
trainloader, testloader, num_examples, X_val, y_val, feature_names = load_data()

class FeverClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}

fl.client.start_numpy_client(server_address="localhost:8080", client=FeverClient())