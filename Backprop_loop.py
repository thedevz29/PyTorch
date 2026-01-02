import torch

inp_x = float(input("Enter x: ")) #taking input of x
inp_y = float(input("Enter y (end result): ")) #taking input of y

x = torch.tensor(inp_x) #input value
y = torch.tensor(inp_y) #pre determined result (only used for training purpose)
lr = 0.001
if x <=50 and y <=75:
    lr  = 0.001
elif x <=75 and y<=100:
    lr = 0.01
elif x >= 100 and y >=150:
    lr = 0.1
else:
    pass
w = torch.tensor(1.0, requires_grad=True) #weight

epoch = 1
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
    print("\n\n")
    
    w.data -= lr*w.grad
    w.grad.zero_()
    
    epoch += 1
    if epoch == 1000:
        print("Maximum epoches reached, terminating loop")
        break 
