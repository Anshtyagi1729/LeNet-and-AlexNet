import torch 
import torch.nn.functional as F
from torch import optim

def train(model,train_loader,test_loader,device,epochs=10,lr=0.01):
    model.to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    for epoch in range(epochs):
        print(f"training on the epoch -> {epoch+1}")
        model.train()
        total_loss=0
        for X , y in train_loader:
            X,y = X.to(device),y.to(device)
            y_hat=model(X)
            loss=F.cross_entropy(y_hat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        print(f"epoch {epoch+1}, Loss: {total_loss:.4f}")
        test(model,test_loader,device)
def test(model,test_loader,device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Test Accuracy: {100.0 * correct / total:.2f}%")     