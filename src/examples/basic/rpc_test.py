import torch
import torch.distributed.autograd as autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn

import os
import time


class MyModule(nn.Module):
    def __init__(self, device, comm_mode):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(1000, 1000).to(device)
        self.comm_mode = comm_mode

    def forward(self, x):
        # x.to() is a no-op if x is already on self.device
        y = self.linear(x.to(self.device))
        return y.cpu() if self.comm_mode == "cpu" else y

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]


def measure():

    lm = MyModule("cpu", "cpu")
    rm = rpc.remote("worker0", MyModule, args=('cuda:0', "gpu"))
    x = torch.randn(1000, 1000).cpu()

    for _ in range(10):
        with autograd.context() as ctx:
            y = rm.rpc_sync().forward(lm(x))
            autograd.backward(ctx, [y.sum()])

    print(lm.named_parameters().__next__())


def run_worker(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

    if rank == 0:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=5
        )
    else:
        options.set_device_map("worker0", {"cpu": 0})
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=5,
            rpc_backend_options=options
        )
        measure()

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    world_size = 5
    mp.spawn(run_worker, nprocs=world_size, join=True)