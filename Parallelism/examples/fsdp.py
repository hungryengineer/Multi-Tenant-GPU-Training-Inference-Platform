import torch
import torch.nn as nn
import torch.distributed as dist

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


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

    # Two CPU ranks form a 1-D FSDP shard mesh.
    mesh = init_device_mesh("cpu", (world_size,))

    model = TinyModel()

    # FSDP2 should be applied bottom-up.
    fully_shard(model.layer1, mesh=mesh)
    fully_shard(model.layer2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    print(f"\n=== RANK {rank} ===")

    for name, param in model.named_parameters():
        print(
            f"{name}: "
            f"type={type(param).__name__}, "
            f"global_shape={tuple(param.shape)}, "
            f"local_shape={tuple(param.to_local().shape)}, "
            f"placements={param.placements}"
        )

    # Now force the first layer to materialize its full parameters.
    model.layer1.unshard()

    print(f"\n=== RANK {rank} AFTER layer1.unshard() ===")

    for name, param in model.layer1.named_parameters():
        print(
            f"{name}: "
            f"type={type(param).__name__}, "
            f"shape={tuple(param.shape)}"
        )

    model.layer1.reshard()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()