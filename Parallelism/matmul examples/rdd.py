import torch
import torch.distributed as dist


def main() -> None:
    # Use gloo so the test works even when GPU memory is occupied
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([float(rank + 1)], device=device)

    print(
        f"backend={backend} device={device} "
        f"rank={rank} world_size={world_size} "
        f"before={tensor.item()}",
        flush=True,
    )

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(
        f"backend={backend} device={device} "
        f"rank={rank} world_size={world_size} "
        f"after={tensor.item()}",
        flush=True,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()