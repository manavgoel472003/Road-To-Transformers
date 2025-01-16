import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train_model(model: torch.nn.Module, X, Y, batch_size=128, lr=5e-4, epochs=300000, plot_loss=False, device="cpu", save_path_name=None):
    lossi = []
    model.to(device);
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        ix = torch.randint(0, X.shape[0], (batch_size, ))
        Xb, Yb = X[ix].to(device), Y[ix].to(device)
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 10000 == 0:
            print(f"{ep:7d} / {epochs:7d} : {loss.item():.4f}")

        lossi.append(loss.log10().item())
    
    if plot_loss:
        plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
        plt.title("Mean of Log10(Loss) for 1000 iterations")
        plt.ylabel("Loss")
        plt.show()

    if save_path_name: torch.save(model.state_dict(), save_path_name)
    return model

@torch.no_grad()
def evaluate_split(model: torch.nn.Module, x, y, device="cpu"):
    model.to(device);
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(f"Loss : {loss.item()}")
