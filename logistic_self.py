import torch
import torch.nn as nn
from sklearn import datasets as ds
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split as tts

ds = ds.load_breast_cancer()
x,y = ds.data, ds.target

n_samples, n_features = x.shape

x_train, x_test, y_train, y_test = tts(x,y, test_size = 0.3, random_state = 123)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#tensor conversion
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#reshape y
y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

#MAKING THE MODEL WITH NN.MODULE PARENT CLASS (inheritance)
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        
    def forward(self, x):
        y_prediction = torch.sigmoid(self.linear(x))
        return y_prediction
    
    
#running an instance
model = LogisticRegression(n_features)

lr = 0.1
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
epochs = 2500

# training loop
for epoch in range(epochs):
    # Forward pass
    y_predicted = model(x_train)
    loss_value = loss(y_predicted, y_train)
    
    # Backward pass and optimization
    loss_value.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_value.item():.4f}')

# Evaluation
with torch.no_grad():
    y_train_pred = model(x_train)
    y_train_pred_cls = y_train_pred.round()
    train_acc = (y_train_pred_cls == y_train).sum().item() / y_train.shape[0]
    
    y_test_pred = model(x_test)
    y_test_pred_cls = y_test_pred.round()
    test_acc = (y_test_pred_cls == y_test).sum().item() / y_test.shape[0]
    
    print(f'Train Accuracy: {train_acc * 100:.2f}%')
    print(f'Test Accuracy: {test_acc * 100:.2f}%')

