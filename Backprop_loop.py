import torch

inp_x = float(input("Enter x: "))
inp_y = float(input("Enter y (end result): "))

x = torch.tensor(inp_x) #input value
y = torch.tensor(inp_y) #pre determined result (only used for training purpose)
lr = 0.001
w = torch.tensor(1.0, requires_grad=True) #weight

epoch = 0
while True:
    print("Iteration number: ", epoch)
    y_est = w*x
    print("Estimated y = ", y_est)
    loss = (y_est-y)**2
    print("Loss = ", loss)
    
    
    if loss.item() < 0.001:
        print("Satisfactory preiction reached!! Stopping at epoch :", epoch)
        break
    loss.backward()
    print("w.grad = ", w.grad)
    
    w.data -= lr*w.grad
    w.grad.zero_()
    
    epoch += 1
    if epoch == 1000:
        print("Maximum epoches reached, terminating loop")
        break 
