import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 8, bias=False)
        self.layer2 = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.layer2(self.layer1(x))


def main():
    dist.init_process_group("gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    mesh = init_device_mesh("cpu", (world_size,))

    model = TinyModel()

    fully_shard(model.layer1, mesh=mesh)
    fully_shard(model.layer2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(2, 8)
    target = torch.randn(2, 4)

    output = model(x)
    loss = nn.functional.mse_loss(output, target)

    print(f"RANK {rank} loss={loss.item():.6f}")

    loss.backward()

    print(f"\n=== RANK {rank} GRADIENTS ===")

    for name, param in model.named_parameters():
        print(
            f"{name}: "
            f"type={type(param).__name__}, "
            f"shape={tuple(param.shape)}, "
            f"grad_shape={None if param.grad is None else tuple(param.grad.shape)}"
        )

    optimizer.step()

    print(f"RANK {rank} optimizer step complete")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()