from torch import nn
import torch.quantization as tq

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = tq.QuantStub()
        self.seq = nn.Sequential(nn.Linear(16, 16))
        self.linear = nn.Linear(16, 16)
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.seq(x)
        x = self.linear(x)
        x = self.dequant(x)
        return x
    
