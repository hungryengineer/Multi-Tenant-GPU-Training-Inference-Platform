import torch
import torch.distributed as dist


def main():
    dist.init_process_group("gloo", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # X = [1, 2]
    X = torch.tensor([[1.0, 2.0]])

    # Full W = [2 x 4]
    W = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ])

    # Row-wise split
    W_local = torch.chunk(W, world_size, dim=0)[rank]

    # Each rank gets the corresponding portion of X
    X_local = X[:, rank:rank + 1]

    # Local partial result
    Y_local = X_local @ W_local

    print(
        f"rank={rank} "
        f"X_local={X_local.tolist()} "
        f"W_local={W_local.tolist()} "
        f"partial={Y_local.tolist()}"
    )

    # Sum partial results across ranks
    dist.all_reduce(Y_local, op=dist.ReduceOp.SUM)

    print(
        f"rank={rank} "
        f"final={Y_local.tolist()}"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()