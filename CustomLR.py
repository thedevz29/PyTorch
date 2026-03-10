#import frameworks
import torch
import torch.nn as nn

x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

n_samples, n_features = x.shape

input_size = n_features
output_size = n_features

test = torch.tensor([5], dtype= torch.float32)#test tensor

# model = nn.Linear(input_size, output_size) - this can also be used but the class defined below is custom model

class LinearRegression(nn.Module): #custom Linear Regression model
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self,x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)
print(f"Prediction before training f(test) = {model(test).item():.3f}")

lr = 0.1
limit = 300
loss = nn.MSELoss() #manual loss calculation can also be used
optimizer = torch.optim.SGD(model.parameters(), lr = lr) #manual optimizer can also be used


for epoch in range(limit+1):
    
    pred = model(x)
    
    l = loss(y, pred)
    
    l.backward()
    
    optimizer.step() #optimizer functions
    
    optimizer.zero_grad()
    
    
    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch}: w = {w[0][0].item(): .3f} loss = {l: .8f}')

print(f"Prediction after training f(5) = {model(test).item():.3f}")