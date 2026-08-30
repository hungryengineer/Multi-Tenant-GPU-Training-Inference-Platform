# Multi-Tenant GPU Training & Inference Platform

Welcome to the **Multi-Tenant GPU Platform**, a highly optimized Kubernetes-native architecture designed to schedule, execute, and serve distributed GPU workloads. 

## 🎯 The End Goal: Dynamic vLLM LoRA Serving
The core objective of this platform is to provide a seamless, end-to-end pipeline for **Multi-Tenant Large Language Models (LLMs)**. We achieve this by:
1. **Containerized Fine-Tuning**: Running highly-efficient QLoRA fine-tuning jobs (via Unsloth) scheduled through **Kueue** and executed on a **KubeRay** cluster.
2. **Persistent Checkpointing**: Saving the resulting LoRA adapters to persistent storage (PVCs) shared across the cluster.
3. **Dynamic Inference**: Passing the generated LoRA artifacts directly into **vLLM** via a Kubernetes Gateway API (Envoy). This allows a single massive base model to serve multiple concurrent users, seamlessly switching out lightweight LoRA adapters per request without loading identical base models into VRAM.

## 🏗️ Architecture Overview

The platform isolates Training from Inference while bridging them through persistent storage artifacts.

```text
                         K3s (Kubernetes)
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

## 🚀 Implementation Guide

### 1. The Training Pipeline (KubeRay + Kueue)
The training pipeline handles job queuing and distributed GPU execution.
* **Kueue** acts as a strict batch scheduler, ensuring multiple jobs don't thrash GPU memory by running concurrently.
* **KubeRay** spins up ephemeral Ray clusters (Head and Worker pods) exactly when Kueue grants quota.
* **Unsloth** runs inside the Ray Worker, applying 2x faster bfloat16 LoRA patching to models like `Qwen-0.5B`.
* **Artifacts** are streamed into a `local-path` Persistent Volume Claim (PVC), allowing checkpoints to persist even if pods are preempted.

### 2. The Inference Pipeline (vLLM + Envoy Gateway)
*Once training completes, the pipeline transitions to inference.*
* **Gateway API (EPP/Envoy)** routes incoming tenant REST requests to the correct model.
* **vLLM** loads the singular base model into GPU memory. 
* As requests arrive for different fine-tuned tasks, vLLM dynamically fetches the corresponding LoRA adapters generated from the training phase, injecting them on-the-fly with near-zero latency overhead.

---

## 🛠️ Quick Start

**Prerequisites:** A local K3s cluster with NVIDIA device plugins, Kueue, and the KubeRay operator installed.

### Step 1: Build the Docker Image
We utilize a heavily optimized two-stage Docker environment using `uv` to minimize layer sizes and build times.
```bash
docker build -f Dockerfile.ray -t mt-gpu-training-ray:dev training/
```

### Step 2: Bridge to K3s Containerd
Because K3s does not natively read the local Docker daemon cache, you must export the image and import it into containerd's `k8s.io` namespace.
```bash
docker save mt-gpu-training-ray:dev > mt-gpu-training-ray.tar
sudo k3s ctr --namespace k8s.io images import mt-gpu-training-ray.tar
rm mt-gpu-training-ray.tar # Free up space!
```

### Step 3: Initialize Persistent Storage
Create the PVC to hold your LoRA checkpoints:
```bash
kubectl apply -f qlora-pvc.yml
```

### Step 4: Submit the RayJob
Submit the job to Kueue. The Ray operator will automatically spawn the cluster, inject the `uv run python train_qlora.py` entrypoint, and save the final adapters to `/app/output`.
```bash
kubectl apply -f qlora-init.yml
```

---

## 🧠 Key Technical Decisions & Optimizations

* **Garbage Collection Evasion:** Large ML images (>12GB) easily trigger Kubernetes `DiskPressure`. We aggressively prune docker caches and use direct `tar` imports to keep Kubelet from evicting our Kueue scheduler.
* **Non-Root Execution:** We override default HuggingFace outputs (which target `/output`) by injecting `OUTPUT_DIR=/app/output` via the RayJob entrypoint, ensuring our non-root `ray` user never encounters PermissionDenied errors.
* **Seamless Resumption:** The `SFTTrainer` natively detects the mounted PVC. If a pod crashes, the next scheduled pod will instantly resume from the latest `checkpoint-X` folder without losing epochs.
* **Bfloat16 Native:** Enforced `bf16=True` throughout the stack to align with modern model precision targets (Qwen/Llama), avoiding Triton compilation crashes and type mismatches.
