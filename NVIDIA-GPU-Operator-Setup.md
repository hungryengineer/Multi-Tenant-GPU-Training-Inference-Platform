# NVIDIA GPU Operator Setup

This document consolidates the commands used to set up NVIDIA GPU support in the Kubernetes (K3s) cluster for the Multi-Tenant GPU platform. It covers installing the **NVIDIA GPU Operator** (which includes the NVIDIA device plugin) and cleaning up the legacy / standalone device plugin that was previously running.

The GPU Operator is the recommended, self-managed way to deploy the NVIDIA device plugin, DCGM exporter, MIG manager, and other components as a single unit. It replaces the older standalone "nvidia-device-plugin" Helm chart and keeps everything in a consistent, upgradeable state.

---

## 1. Verify Hardware & Kubernetes Readiness

Before installing anything, confirm NVIDIA drivers are present on the host and that the nodes can report GPU capacity.

```bash
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
kubectl get nodes -o wide
kubectl describe node pop-os | grep -A10 -E "Capacity:|Allocatable:"
```

- `nvidia-smi` confirms the driver is loaded and the GPU is visible on the host.
- The node `Capacity` / `Allocatable` block is where the device plugin will later advertise `nvidia.com/gpu`.

---

## 2. Add the NVIDIA Helm Repository

The GPU Operator and the device plugin charts are both distributed via NVIDIA's Helm repository.

```bash
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update
```

Adds the NVIDIA `gpu-operator` chart repository and refreshes the local Helm index.

---

## 3. Install the NVIDIA GPU Operator

```bash
helm install gpu-operator nvidia/gpu-operator \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true
```

Installs the full GPU Operator stack. The operator then deploys the **NVIDIA device plugin** as a DaemonSet that:

- Detects GPUs on each node.
- Registers the `nvidia.com/gpu` extended resource with the Kubelet.
- Makes the GPUs schedulable by Kubernetes so pods can request them via `resources.limits: nvidia.com/gpu: 1`.

Enabling the **DCGM exporter** optionally exposes GPU metrics (temperature, utilization, power, memory) for monitoring.

> **Note:** In this environment the device plugin was installed via its standalone Helm chart into the `nvidia-device-plugin` namespace with release name `nvdp`. Both approaches deploy the same underlying DaemonSet; the GPU Operator bundles them together.

---

## 4. Remove the Old Standalone NVIDIA Device Plugin

Before (or after) installing the GPU Operator, the legacy device plugin must be removed to avoid two DaemonSets racing to advertise the same `nvidia.com/gpu` resource.

### 4a. Inspect what is currently deployed

```bash
helm list -A                      # see all Helm releases across namespaces
helm list -A | grep -i nvdp       # confirm the old device-plugin release
helm get values nvdp -n nvidia-device-plugin

kubectl get pods -A | grep -Ei 'nvidia-device-plugin|gpu-operator|dcgm'
kubectl -n nvidia-device-plugin get daemonset nvdp-nvidia-device-plugin -o yaml
kubectl -n nvidia-device-plugin get configmap nvdp-nvidia-device-plugin-configs -o yaml
```

These commands confirm whether the old device plugin is present and how it was configured before tearing it down.

### 4b. Uninstall the old device plugin

```bash
helm uninstall nvdp -n nvidia-device-plugin
```

Removes the standalone device plugin release, which deletes its DaemonSet pods from the node. The node will temporarily drop its `nvidia.com/gpu` capacity until the GPU Operator's device plugin comes up.

Optionally, if you need to clean up the namespace itself:

```bash
kubectl delete namespace nvidia-device-plugin --ignore-not-found
```

### 4c. Verify the GPU Operator device plugin is serving

```bash
kubectl get pods -A | grep -Ei 'nvidia-device-plugin|gpu-operator|dcgm'
kubectl get ds -A | grep -Ei 'gpu|nvidia|dcgm'

kubectl logs -n nvidia-device-plugin \
  $(kubectl get pods -n nvidia-device-plugin -l app.kubernetes.io/name=nvidia-device-plugin \
    -o jsonpath='{.items[0].metadata.name}') \
  -c nvidia-device-plugin-ctr | grep -Ei 'config|timeSlicing|Starting to serve|registered'
```

You should see the DaemonSet pods reporting `Starting to serve` and the node advertising `nvidia.com/gpu`.

---

## 5. Confirm the GPU Resource is Schedulable

```bash
kubectl get node pop-os -o jsonpath='{.status.capacity.nvidia\.com/gpu}{"\n"}'
kubectl get node pop-os -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
kubectl describe node pop-os | grep -A20 "Allocated resources:"
```

The node should now report a non-zero `nvidia.com/gpu` value in both `Capacity` and `Allocatable`, meaning Kubernetes can schedule GPU workloads onto it.

---

## 6. Option: GPU Time-Slicing (Temporal Sharing)

To let multiple pods share a single physical GPU, configure **time-slicing** via a ConfigMap in the device plugin namespace.

```bash
kubectl edit configmap nvdp-nvidia-device-plugin-configs -n nvidia-device-plugin
```

A time-slicing config typically looks like:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nvdp-nvidia-device-plugin-configs
  namespace: nvidia-device-plugin
data:
  time-slicing: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4
```

After editing, restart the DaemonSet so it picks up the new configuration:

```bash
kubectl rollout restart ds -n nvidia-device-plugin <device-plugin-daemonset>
```

> The GPU Operator manages its device plugin ConfigMap automatically; if you are using the operator, apply time-slicing through the operator's `ClusterPolicy` (`.spec.devicePlugin.config`) instead of hand-editing the ConfigMap.

---

## 7. Validation / Smoke Tests

Verify GPU scheduling end to end by running a small test workload that requests a GPU.

```bash
kubectl run gpu-test \
  --image=nvidia/cuda:12.8.1-base-ubuntu24.04 \
  --restart=Never \
  --limits='nvidia.com/gpu=1' \
  --command -- nvidia-smi

kubectl get pod gpu-test -o wide
kubectl logs gpu-test
kubectl exec gpu-test -- nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
kubectl delete pod gpu-test --ignore-not-found
```

If the pod runs `nvidia-smi` successfully, the device plugin is correctly exposing the GPU to the cluster. You can repeat this with several pods simultaneously to confirm time-slicing (if configured) and to stress the scheduler:

```bash
for i in 1 2 3; do
  kubectl run gpu-test-$i \
    --image=nvidia/cuda:12.8.1-base-ubuntu24.04 \
    --restart=Never \
    --limits='nvidia.com/gpu=1' \
    --command -- nvidia-smi
done
```

---

## Summary of Key Commands

```bash
# 1. Pre-flight checks
nvidia-smi
kubectl get nodes -o wide
kubectl describe node pop-os | grep -A10 -E "Capacity:|Allocatable:"

# 2. Add Helm repo
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

# 3. Install the GPU Operator
helm install gpu-operator nvidia/gpu-operator \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true

# 4. Remove the old standalone device plugin
helm list -A | grep -i nvdp
helm uninstall nvdp -n nvidia-device-plugin
kubectl delete namespace nvidia-device-plugin --ignore-not-found   # optional

# 5. Verify device plugin + GPU resource
kubectl get pods -A | grep -Ei 'nvidia-device-plugin|gpu-operator|dcgm'
kubectl get node pop-os -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'

# 6. (Optional) Time-slicing config
kubectl edit configmap nvdp-nvidia-device-plugin-configs -n nvidia-device-plugin
kubectl rollout restart ds -n nvidia-device-plugin <daemonset>

# 7. Smoke test
kubectl run gpu-test --image=nvidia/cuda:12.8.1-base-ubuntu24.04 \
  --restart=Never --limits='nvidia.com/gpu=1' --command -- nvidia-smi
```

---

## Notes & Gotchas

- **Do not run two device plugins at once.** The GPU Operator's DaemonSet and the standalone `nvdp` release both register the same `nvidia.com/gpu` resource. Uninstall the old one before (or immediately after) installing the operator, otherwise the scheduler can see conflicting capacities.
- **Image pulls from Docker vs. Containerd (K3s):** K3s does not read the local Docker daemon cache. Export and import GPU/CUDA images if you build them locally:
  ```bash
  docker save <image> > <image>.tar
  sudo k3s ctr --namespace k8s.io images import <image>.tar
  ```
- **GPU memory is not oversubscribed by default.** Time-slicing shares compute cycles, not VRAM. Multiple large models on one GPU can still hit out-of-memory.
- Non-root pods won't have driver access unless the device plugin mounts the driver libraries and `nvidia.com/gpu` is requested in pod limits.
