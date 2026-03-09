import torch

i_x = int(input("Enter the value of tensor x : "))
i_y = int(input("Enter the value of tensor y : "))

x = torch.tensor(float(i_x))
y = torch.tensor(float(i_y))

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)

lr = 0.01

for epoch in range(1000):

    prediction = w * x + b
    loss = (prediction - y)**2

    if loss.item() < 0.001:
        print("Expected result reached!! at epoch number - ", epoch)
        break

    loss.backward()

    w.data -= lr * w.grad
    b.data -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

print("Final weight:", w.item())
print("Final bias:", b.item())
 