"""Custom Linear Regression Model using PyTorch."""

import torch
import torch.nn as nn


# Data preparation
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = x.shape

input_size = n_features
output_size = n_features

# Test tensor for prediction
test = torch.tensor([5], dtype=torch.float32)


class LinearRegression(nn.Module):
    """Custom Linear Regression model."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize the Linear Regression model.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.lin(x)


# Model initialization
model = LinearRegression(input_size, output_size)
print(f"Prediction before training: f(5) = {model(test).item():.3f}")

# Training configuration
learning_rate = 0.1
epochs = 300
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs + 1):
    # Forward pass
    predictions = model(x)

    # Calculate loss
    loss = loss_fn(y, predictions)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    # Log progress
    if epoch % 10 == 0:
        weight, bias = model.parameters()
        print(f"Epoch {epoch:3d}: weight = {weight[0][0].item():7.3f}, loss = {loss.item():.8f}")

# Final prediction
print(f"Prediction after training: f(5) = {model(test).item():.3f}")