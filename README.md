# Multi-Tenant-GPU-Training-Inference-Platform

## Project Goal
To build a highly optimized, multi-tenant Kubernetes architecture using **KubeRay** and **Kueue** capable of scheduling and executing distributed GPU workloads. The initial objective was to successfully containerize and deploy a Qwen-0.5B QLoRA fine-tuning workflow (using Unsloth) as a KubeRay `RayJob` on a local K3s cluster.

## Architecture

```text
                         K3s
                          │
              ┌───────────┴───────────┐
              │                       │
           TRAINING                INFERENCE
              │                       │
           Kueue                     ┌┴────────────┐
              │                       │             │
            RayJob              Gateway/EPP       llm-d
              │                       │             │
            QLoRA                     └──────┬──────┘
              │                              │
        LoRA artifacts                     vLLM
              │                              │
              └──────────────────────→  Multi-LoRA
                                             │
                                           GPU
```

---

## Installation Sequence
1. Kueue
2. Ray (KubeRay Operator)

---

## Steps Completed

1. **Docker Standardization & Optimization**
   - Refactored `Dockerfile.cuda` and `Dockerfile.ray` to safely manage permissions for non-root execution (switching to the `ray` user) while maintaining system dependency parity.
   - Drastically reduced Docker build times by creating a `.dockerignore` file, dropping the build context transfer from over 5GB (due to `.venv` and caches) down to ~500KB.
2. **KubeRay Manifest Refinement**
   - Restructured the `RayJob` YAML definitions (`qlora-init.yml`, `ray-b.yml`) to correctly utilize KubeRay standards.
   - Aligned Python environments across head and worker nodes to prevent cluster segmentation.
3. **Local Registry & Containerd Integration**
   - Successfully bridged the gap between the local Docker daemon and K3s's `containerd` runtime by exporting and importing image tarballs directly into the `k8s.io` namespace.
4. **End-to-End Validation**
   - Deployed the `qlora-test` job to K3s. 
   - Actively monitored the Ray job submitter and worker nodes, verifying that Unsloth dynamically patched the model, tokenized the Alpaca dataset, and successfully completed 125 training steps on the RTX 5050.
5. **Git Hygiene**
   - Added a `.gitignore` to prevent massive archives (`.tar`) and generated Python/Unsloth cache files from polluting the repository.

---

## Issues Faced & Resolutions

1. **Triton Compiler Crash**
   - **Issue:** Unsloth crashed immediately because Triton could not compile custom CUDA kernels.
   - **Fix:** Added `build-essential` (gcc/g++) to the Dockerfiles as a root user before switching to the `ray` user.
2. **Precision Mismatch (TypeError)**
   - **Issue:** Training crashed due to `fp16=True` being applied to a base model natively trained in bfloat16.
   - **Fix:** Modified the `SFTTrainer` arguments in `train_qlora.py` to use `bf16=True` instead.
3. **YAML Strict Decoding Errors**
   - **Issue:** `kubectl apply` rejected the KubeRay manifest with `unknown field "spec.workerGroupSpecs"`.
   - **Fix:** Corrected the YAML indentation, placing `workerGroupSpecs` correctly under `rayClusterSpec`.
4. **K3s ImagePullBackOff**
   - **Issue:** Worker pods were stuck trying to pull `mt-gpu-training-ray:dev` from Docker Hub because K3s (containerd) does not share the local Docker image cache.
   - **Fix:** Exported the image (`docker save`) and manually imported it into containerd: `sudo k3s ctr --namespace k8s.io images import mt-gpu-training-ray.tar`.
5. **Ray Operator Bootstrapping Failures**
   - **Issue:** The job submitter failed with `Error: Missing argument 'ENTRYPOINT'`, and the cluster nodes just idled.
   - **Fix:** Added the mandatory `spec.entrypoint` field to the YAML and removed hardcoded `command: ["bash", "sleep"]` overrides from the containers so the KubeRay operator could inject the necessary `ray start` commands.
6. **Strict Python Version Mismatches**
   - **Issue:** The Raylet worker crashed with `Version mismatch: The cluster was started with Python 3.9... This process started with Python 3.12`.
   - **Fix:** Updated the `ray-head` container in the YAML to use our custom `mt-gpu-training-ray:dev` image to ensure absolute environment parity between the head and worker nodes.
7. **Root Filesystem Permission Denied**
   - **Issue:** The training script successfully finished but crashed when attempting to save the LoRA adapters, throwing `PermissionError: [Errno 13] Permission denied: '/output'`. 
   - **Fix:** Injected `env OUTPUT_DIR=/app/output/lora-adapter` directly into the RayJob `entrypoint` command, directing the script to save the artifacts in the non-root `/app` directory where it had write access.
