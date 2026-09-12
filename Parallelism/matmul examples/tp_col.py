import torch
import torch.distributed as dist


def main():
    dist.init_process_group("gloo", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    X = torch.tensor([[1.0, 2.0]])

    # Full weight matrix would be [2 x 4]
    W = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ])

    # Column-wise tensor parallel split
    W_local = torch.chunk(W, world_size, dim=1)[rank]

    Y_local = X @ W_local

    print(
        f"rank={rank} "
        f"W_local={W_local.tolist()} "
        f"Y_local={Y_local.tolist()}"
    )

    dist.barrier()

    # Gather the output pieces
    gathered = [torch.zeros_like(Y_local) for _ in range(world_size)]
    dist.all_gather(gathered, Y_local)

    if rank == 0:
        Y = torch.cat(gathered, dim=1)
        print(f"Final Y = {Y.tolist()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()