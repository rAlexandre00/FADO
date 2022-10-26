import torch
import torch.distributed.rpc as rpc
import torch.distributed.autograd as autograd
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(1000, 1000).to(device)

    def forward(self, x):
        y = self.linear(x.to(self.device))
        return y.cpu()

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]

options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
options.set_device_map("worker1", {"cpu": 0})
rpc.init_rpc("worker0", rank=0, world_size=3, rpc_backend_options=options)
lm = MyModule("cpu")
rm = rpc.remote("worker1", MyModule, args=('cuda:0',))
x = torch.randn(1000, 1000).cpu()

for _ in range(10):
    with autograd.context() as ctx:
        y = rm.rpc_sync().forward(lm(x))
        autograd.backward(ctx, [y.sum()])

#torch.cuda.current_stream("cuda:0").synchronize()
print(lm.named_parameters().__next__())
rpc.shutdown()
