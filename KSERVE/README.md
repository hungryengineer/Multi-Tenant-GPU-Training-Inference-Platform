# KServe LLM Inference — LLMInferenceService (vllm-openai)

Operator-oriented guide for bringing up `qwen-0.5b` with KServe's
`LLMInferenceService` (`llmisvc-controller-manager`, KServe **v0.17.0**) on the SAME
single 8 GiB GPU that is also running the Ray stack (see `../RAY/README.md`).

Everything below was verified on this node (single-node k3s, one RTX 5050 7.53 GiB,
fully offline / air-gapped images).

---

## 1. Prerequisites (verified state on this node)

- `k3s` single node, GPU schedulable via `nvidia.com/gpu`.
- KServe installed from the git source checkout in this folder (`KSERVE/kserve/`):
  - controller `llmisvc-controller-manager` in the `kserve` namespace (v0.17.0)
  - admission webhooks for `LLMInferenceService` (validate + default)
  - Envoy gateway `kserve-ingress-gateway` (see pitfall §4.6 — NOT reachable here)
- Local image cached in embedded containerd: `vllm/vllm-openai:v0.27.1`.
- The model snapshot already on the host: `/var/lib/qwen2.5-0.5b-instruct`
  (mounted via `hostPath`).
- StorageInitializer is DISABLED in the manifest — no image pull, no download, fully
  offline (see §4.8).
- Manifest: `KSERVE/kserve-qwen.yml` (apiVersion `serving.kserve.io/v1alpha1`;
  the CRD also serves `v1alpha2`).

## 2. Apply

```bash
kubectl apply -f KSERVE/kserve-qwen.yml -n default
```

The controller synthesizes a normal Kubernetes Deployment named
`qwen-llm-kserve` and — critically — it **injects the vLLM command itself**:

```text
vllm serve /mnt/models --served-model-name qwen --port 8000
```

Your `template.containers[].args` are appended as flags AFTER that positional path.

Watch it come up:

```bash
kubectl get llminferenceservice qwen-llm -n default -o wide
kubectl get deploy qwen-llm-kserve -n default
kubectl get pods -n default -l app.kubernetes.io/name=qwen-llm --show-labels
kubectl rollout status deploy/qwen-llm-kserve -n default
```

Expected steady state: single pod `qwen-llm-kserve-<hash>-<pod>` `Running`, Ready=1.

## 3. Verify serving

The pod has no curl — use python `urllib` from inside it (or from any node pod):

```bash
kubectl exec <qwen-llm-kserve-pod> -- python3 - <<'PY'
import json, urllib.request
print("health:", urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10).read()[:60])
body = json.dumps({
    "model": "qwen",
    "messages": [{"role": "user", "content": "What is the capital of Ethiopia?"}],
    "max_tokens": 20,
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=body, headers={"Content-Type": "application/json"}, method="POST",
)
with urllib.request.urlopen(req, timeout=60) as r:
    print(json.load(r)["choices"][0]["message"]["content"])
PY
# expected: "The capital of Ethiopia is Addis Ababa."
```

Also reachable over the headless ClusterIP the controller creates:

```bash
kubectl get svc -l app.kubernetes.io/name=qwen-llm -n default   # qwen-llm-kserve-workload-svc :8000
```

## 4. Pitfalls (each one cost a long debugging session)

### 4.1 NEVER pass a positional model path in `args`

The controller already injects `vllm serve /mnt/models ...`. vLLM v0.27.1 treats a
second positional argument as a hard CLI error and crash-loops:

```text
vllm: error: unrecognized arguments: /mnt/models
```

`args` must contain **flags only**. Observed-good set:

```yaml
args:
- "--max-model-len"          # "2048"
- "--tensor-parallel-size"   # "1"
- "--gpu-memory-utilization" # "0.40"
- "--no-enable-flashinfer-autotune"
- "--enforce-eager"
```

### 4.2 3 GiB memory limit = silent OOMKilled (exit 137)

The engine allocates weights (~1 GiB) + CUDA state + KV *after* graph setup; with the
pod limited to 3 GiB the container is OOM-killed shortly after "engine started" and
crash-loops healthily. Use `limits.memory: 8Gi` (requests `1400Mi`).

### 4.3 `gpu-memory-utilization` is not "VRAM you gave it"

vLLM keeps ~2.4 GiB of fixed overhead (weights + torch + graphs) before the KV cache.
On this 7.53 GiB GPU:
- `0.40` → ~1.6–2.7 GiB KV cache, actual engine footprint ≈ 2.2–3.6 GiB — matches the
  ~2.2–3.5 GiB measured while coexisting with Ray (Ray holds ~3.4 GiB).
- Leave headroom! At `0.40` total GPU used ≈ 5.6–7.0 GiB of 7.5. Do not raise it just
  to get "more cache", or the OTHER engine on the same GPU will OOM at boot.

### 4.4 FlashInfer sampler JIT stall under host-RAM pressure (the big one)

Cold boots run a one-time JIT (nvcc) compile of the FlashInfer sampling kernels. On a
host where RAM is tight (our node: k3s + Ray head 5 GiB + Ray worker 9 GiB + KServe),
that compile can take many minutes and exceeds the injected startup-probe window
(10 min `/health`) → the pod gets killed and, because the container filesystem is
ephemeral, the JIT result is lost — loop forever. Logs freeze at:
`Using FlashInfer for top-p & top-k sampling` / `Skipping FlashInfer autotune...`.

Three mitigations, all already in `kserve-qwen.yml`:
1. `--no-enable-flashinfer-autotune` — skips the attention autotuner.
2. `--enforce-eager` — skips `torch.compile` + CUDA graph capture entirely; also uses
   noticeably LESS VRAM, which helps coexistence (§4.3).
3. Mount a hostPath for vLLM's cache dir so one-time JIT/compile caches survive
   restarts. Note from our runs: vLLM's `torch_compile_cache`/autotune files go there,
   but the FlashInfer *sampler* kernel build did not (it lives in the container's
   scratch space), so the volume helps but is not a silver bullet — pair it with the
   two flags and enough host RAM (`sudo free -m`, target `available` > ~3 GiB).

```yaml
volumes:
- name: vllm-cache
  hostPath: { path: /var/lib/vllm-cache }
containers:
- volumeMounts:
  - name: vllm-cache
    mountPath: /home/.cache
```

### 4.5 Single-GPU rollout deadlock

Danger: any rollout that replaces the pod requires the NEW engine to boot while the
OLD pod still holds VRAM → CUDA OOM / `connection refused` forever, neither side moves.
If the GPU is shared with Ray (§4.3), you must break the deadlock by hand:

```bash
kubectl scale rs <old-rs-name> --replicas=0   # frees VRAM, new engine boots
kubectl get rs -l app.kubernetes.io/name=qwen-llm -n default
```

Do this whenever a fresh rollout parks mid-way with the old pod still Ready
(especially with the shared-GPU setup).

### 4.6 Ingress/gateway is NOT usable here

The controller emits an HTTPRoute to the Envoy gateway
`kserve-ingress-gateway` (namespace `kserve`), but on this install that Gateway stays
`Programmed=False`, so the public path (`/default/qwen-llm/v1/*`, `/v1/*`) is **not**
reachable. Use `pod:8000` or the generated Service `qwen-llm-kserve-workload-svc:8000`
instead. The HTTPRoute also tries to reference an `InferencePool` — that migration is
only relevant once a real gateway is in place.

### 4.7 `router.scheduler.pool` spawns extra deployments

With `spec.router.scheduler.pool: {}` the controller generates a side Deployment
`qwen-llm-kserve-router-scheduler-*`. Its image often isn't cached on an air-gapped
node → `ImagePullBackOff` / `Pending`. Harmless: not required for serving through the
workload Service; scale it to 0 or ignore it. Same for any `*-router-scheduler-*` pods.

### 4.8 Offline operation

- `spec.storageInitializer.enabled: false` + `hostPath` model mount = zero downloads;
  the `model.uri` (`hf://Qwen/Qwen2.5-0.5B-Instruct`) is metadata only here.
- `imagePullPolicy: Never` on every image; anything not already cached will hang/fail.

### 4.9 Rollout contention with the Ray side

Because both stacks share the node RAM as well as the GPU, apply only ONE change at a
time and wait for it to settle (see §4.4/§4.5). If a boot looks frozen, check host
`sudo free -m` and `nvidia-smi` first — the fix is almost always "free VRAM / RAM for
the boot", not a config error.

## 5. Troubleshooting cheat-sheet

| Symptom | Cause / fix |
|---|---|
| CrashLoop `unrecognized arguments: /mnt/models` | positional arg in `args`; flags only (§4.1) |
| OOMKilled exit 137 after "engine started" | `limits.memory` too low; use 8 Gi (§4.2) |
| Pod ready never, log at `FlashInfer` line | host RAM pressure; add the 3 mitigations (§4.4) |
| Rollout stuck, old pod Ready, new pod OOM | scale old ReplicaSet to 0 (§4.5) |
| `router-scheduler-*` Pending / ImagePullBackOff | harmless side deployment; scale 0 (§4.7) |
| No external URL works | gateway `Programmed=False`; use pod/ClusterIP (§4.6) |