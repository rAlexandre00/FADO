import torch.distributed.rpc as rpc
import torch.nn as nn


class MyModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(1000, 1000).to(device)

    def forward(self, x):
        y = self.linear(x.to(self.device))
        return y

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]

rpc.init_rpc("worker1", rank=1, world_size=3)
rpc.shutdown()