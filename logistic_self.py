import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts

dataset = datasets.load_breast_cancer()
x, y = dataset.data, dataset.target

n_samples, n_features = x.shape

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.3, random_state=123)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# tensor conversion
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape y
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# MAKING THE MODEL WITH NN.MODULE PARENT CLASS (inheritance)
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        return self.linear(x)  # no sigmoid; using BCEWithLogitsLoss


# running an instance
model = LogisticRegression(n_features)

lr = 0.01
loss = nn.BCEWithLogitsLoss()  # numerically stable; includes sigmoid internally
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 1000

# training loop
for epoch in range(epochs):
    optimizer.zero_grad()           # clear gradients first
    y_predicted = model(x_train)
    loss_value = loss(y_predicted, y_train)
    loss_value.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_value.item():.4f}')

# Evaluation
with torch.no_grad():
    y_train_pred_cls = torch.sigmoid(model(x_train)).round()
    train_acc = (y_train_pred_cls == y_train).sum().item() / y_train.shape[0]

    y_test_pred_cls = torch.sigmoid(model(x_test)).round()
    test_acc = (y_test_pred_cls == y_test).sum().item() / y_test.shape[0]

    print(f'Train Accuracy: {train_acc * 100:.2f}%')
    print(f'Test Accuracy: {test_acc * 100:.2f}%')
