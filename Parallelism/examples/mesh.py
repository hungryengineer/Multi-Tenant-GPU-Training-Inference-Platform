import torch.distributed as dist

TP = 2
PP = 2
DP = 2

def main():
    dist.init_process_group("gloo", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert world_size == TP * PP * DP

    # Mapping used in our example:
    # rank = (((dp * PP) + pp) * TP) + tp

    tp_rank = rank % TP
    pp_rank = (rank // TP) % PP
    dp_rank = rank // (TP * PP)

    # Groups
    tp_group_ranks = [
        dp_rank * TP * PP + pp_rank * TP + tp
        for tp in range(TP)
    ]

    pp_group_ranks = [
        dp_rank * TP * PP + pp * TP + tp_rank
        for pp in range(PP)
    ]

    dp_group_ranks = [
        dp * TP * PP + pp_rank * TP + tp_rank
        for dp in range(DP)
    ]

    print(
        f"global={rank} "
        f"DP={dp_rank} PP={pp_rank} TP={tp_rank} | "
        f"TP_group={tp_group_ranks} "
        f"PP_group={pp_group_ranks} "
        f"DP_group={dp_group_ranks}"
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()