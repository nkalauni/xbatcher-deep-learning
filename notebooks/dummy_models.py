import torch

class Identity(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

class MeanAlongDim(torch.nn.Module):
    def __init__(self, ax):
        super(MeanAlongDim, self).__init__()
        self.ax = ax

    def forward(self, x):
        return torch.mean(x, self.ax)

# Take first n elements along axis
class SubsetAlongAxis(torch.nn.Module):
    def __init__(self, ax, n):
        super(SubsetAlongAxis, self).__init__()
        self.ax = ax
        self.indexer = torch.tensor(list(range(n)))

    def forward(self, x):
        return torch.index_select(x, self.ax, self.indexer)

class ExpandAlongAxis(torch.nn.Module):
    def __init__(self, ax, n_repeats):
        super(ExpandAlongAxis, self).__init__()
        self.ax = ax
        self.n_repeats = n_repeats

    def forward(self, x):
        return torch.repeat_interleave(x, self.n_repeats, dim=self.ax)

class AddAxis(torch.nn.Module):
    def __init__(self, ax):
        super(AddAxis, self).__init__()
        self.ax = ax

    def forward(self, x):
        return torch.unsqueeze(x, self.ax)

if __name__ == "__main__":
    print(__name__)
    input_tensor = torch.arange(125).reshape((5, 5, 5)).to(torch.float32)
    
    mad = MeanAlongDim(-1)
    print("Mean along dim")
    print("Input shape:", input_tensor.shape)
    print("Output shape:", mad(input_tensor).shape)
    print()

    saa = SubsetAlongAxis(1, 3)
    print("Subset along axis")
    print("Input shape:", input_tensor.shape)
    print("Output shape:", saa(input_tensor).shape)
    print()

    eaa = ExpandAlongAxis(1, 3)
    print("Expand along axis")
    print("Input shape:", input_tensor.shape)
    print("Output shape:", eaa(input_tensor).shape)
    print()

    aa = AddAxis(0)
    print("Add axis")
    print("Input shape:", input_tensor.shape)
    print("Output shape:", aa(input_tensor).shape)