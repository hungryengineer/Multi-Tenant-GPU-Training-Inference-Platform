import os
import torch.distributed as dist

from megatron.core import parallel_state


def main():
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
    )

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size == 8, f"Expected 8 ranks, got {world_size}"

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        order="tp-dp-pp",
        create_gloo_process_groups=True,
    )

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    dp_rank = parallel_state.get_data_parallel_rank()

    tp_world = parallel_state.get_tensor_model_parallel_world_size()
    pp_world = parallel_state.get_pipeline_model_parallel_world_size()
    dp_world = parallel_state.get_data_parallel_world_size()

    print(
        f"global={rank} "
        f"DP={dp_rank}/{dp_world} "
        f"PP={pp_rank}/{pp_world} "
        f"TP={tp_rank}/{tp_world}"
    )

    dist.barrier()

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()