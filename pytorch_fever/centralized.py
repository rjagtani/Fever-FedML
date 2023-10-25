import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import r2_score
import sage
import numpy as np

data = pd.read_csv("fever_encoded.csv")
X = data.drop('quantity(kWh)', axis=1)
y = data['quantity(kWh)']
feature_names = list(X.columns)
X = X.values
y = y.values
# # Standardize the features and targets
# scaler_X = StandardScaler().fit(X)
# X = scaler_X.transform(X)
# scaler_y = StandardScaler().fit(y.reshape(-1, 1))
# y = scaler_y.transform(y.reshape(-1, 1)).flatten()

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 2. Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3. Define the neural network, loss, and optimizer
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = NeuralNetwork(20, 256, 64, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs.view(-1), y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validate
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs.view(-1), y_val)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# 5. Test the final model's performance
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs.view(-1), y_test)
    r2 = r2_score(y_test, test_outputs)
    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"R2 Score: {r2:.4f}")
    # Setup and calculate
    numpy_X_test = np.array(X_test)
    numpy_y_test = np.array(y_test)
    imputer = sage.MarginalImputer(model, numpy_X_test[:100])
    estimator = sage.PermutationEstimator(imputer, 'mse')
    sage_values = estimator(numpy_X_test, numpy_y_test, batch_size=4)
    sage_values.plot(feature_names=feature_names)
    fi_dict = dict(zip(feature_names,sage_values.values))
    print(fi_dict)