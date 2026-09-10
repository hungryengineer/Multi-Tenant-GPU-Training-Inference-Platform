# Ray LLM Inference — RayService (kuberay-operator + ray-llm)

Operator-oriented guide for bringing up `qwen-0.5b` as a Ray Serve `LLMServer` via
`RayService`, and — crucially — keeping it healthy on the SAME single 8 GiB GPU that is
also running the KServe stack (see `../KSERVE/README.md`).

Everything below was verified on this node (single-node k3s, one RTX 5050 7.53 GiB).

---

## 1. Prerequisites (verified state on this node)

- k3s single node (`pop-os`), node label-free, GPU schedulable via `nvidia.com/gpu`.
- `kuberay-operator` v1.7.0 installed in `kuberay-system`.
- Local (air-gapped) images, pulled into embedded containerd **once**:
  - `rayproject/ray-llm:2.52.0-py311-cu128` (bundles vLLM 0.11.0 and the LLMServer code)
  - `vllm/vllm-openai:v0.27.1` (used by the KServe stack, see `../KSERVE`)
- The model snapshot already materialized on the host:
  - HF tokenizer/model files under `/var/lib/ray-hf-cache` (the `huggingface` cache dir), mounted into the Ray head and worker via `hostPath`.
- Manifests referenced from `..`:
  - `RAY/rayservice-qwen.yml` — THE working config (below)
  - `RAY/rayservice-demo.yml` — older demo (`rayproject/ray:2.46`), reference only

> Software versions in play: Ray 2.52 on the head/workers, Serve runtime
> `LLMServer`, executor `RayDistributedExecutor`, served model port 8000. GPU budget
> expected: ~3420 MiB on the GPU, leaving ~690 MiB for KServe's engine.

## 2. Apply

```bash
kubectl apply -f RAY/rayservice-qwen.yml -n default
```

Watch progression:

```bash
# RayService + its RayCluster
kubectl get rayservice ray-qwen -n default -o wide
kubectl get rayclusters.ray.io -n default
kubectl get pods -n default -l ray.io/cluster=ray-qwen --show-labels   # head + worker
```

The RayService owns a `RayCluster`; the Serve app starts on top of the head/worker
pods (head runs the ServeController + OpenAiIngress, the worker runs the GPU engine).

Expected steady state:

```text
NAME       ACTIVE RAYCLUSTER    STATUS     RAY SERVE APPLICATION STATUS
ray-qwen   ray-qwen-<hash>      RUNNING    RUNNING
```

## 3. Verify serving

The model id inside the Ray vLLM engine is **`qwen-0.5b`** (from `model` in the
service graph), NOT `qwen` — use it in the request body. The engine (and the raw OpenAI
API) runs inside the **worker** pod on `127.0.0.1:8000`:

```bash
kubectl exec <ray-qwen-gpu-workers-worker-*> -n default -- python3 - <<'PY'
import json, urllib.request
body = json.dumps({
    "model": "qwen-0.5b",
    "messages": [{"role": "user", "content": "Hello there!"}],
    "max_tokens": 32,
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=body, headers={"Content-Type": "application/json"}, method="POST",
)
with urllib.request.urlopen(req, timeout=120) as r:
    print(json.load(r)["choices"][0]["message"]["content"])
PY
# expected: starts with "Hello there!"
```

Also correctly answers e.g. "What is the capital of Ethiopia?" → "Addis Ababa".

> Note: the head pod's `OpenAiIngress` (port 8000) only maps the `/` route in this
> install and returns 404 for `/v1/*`; use the worker's direct port above. Serve
> control-plane state (no curl/ray CLI inside pods, use python3):

```bash
kubectl exec <head-pod> -- python3 -c \
 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8265/api/serve/applications/", timeout=10).read()[:400])'
```

## 4. Pitfalls (read before changing anything)

### 4.1 CPU starvation on the worker breaks the Serve startup

The Serve `LLMServer` creates a composite placement group `{CPU:1} + {GPU:1}`; its
remote `import/initialize` tasks (`num_cpus=1`, `soft=False`) must land somewhere.
If the worker group only gets the GPUs (default `2` CPUs), those download/init tasks
can't run and the app hangs forever at
`Running <n> tasks to download model files on worker nodes...`.

Fix (already in `rayservice-qwen.yml`):
- head: `headGroupSpec.template.spec.containers[0].resources` CPUs ≥ **4**;
- worker: `workerGroupSpecs[0].template.spec.containers[0].resources` CPUs **4** AND
  `workerGroupSpecs[0].rayStartParams` must include `num-cpus: "4"` — the worker group
  does NOT inherit the head's CPU count, and without the explicit `rayStartParams` you
  get the default 2 and the stall.

### 4.2 The raylet memory monitor kills the engine "for real"

The raylet's REUSABLE memory monitor watches system-wide memory and kills tasks once
the node exceeds ~95 % of *machine* RAM — even when the pod's cgroup looks fine. Symptom:

```text
ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory
```

Disable it via container env on BOTH head and worker (already in the manifest):

```yaml
- name: RAY_memory_monitor_refresh_ms
  value: "0"
- name: RAY_memory_usage_threshold
  value: "1.0"
```

Note: `RAY_MEMORY_MONITOR_ENABLED=0` is the wrong knob and does not stop the kills.
Make sure the memory bump below accompanies this so the cluster stays honest with k8s.

### 4.3 vLLM engine memory budget on an 8 GiB GPU

`EngineConfig.max_worker_memory_utilization` is the fraction of the *visible* GPU the
engine targets. vLLM fixes ~2.4 GiB of overhead (weights + graphs + CUDA state) before
any KV cache; at 0.25 the budget goes NEGATIVE and the engine aborts:

```text
ValueError: No available memory for the cache blocks
```

- 0.25 → dead (−0.5 GiB KV)
- floor ≈ 0.35; **0.38 is the verified value** (+0.47 GiB KV ≈ 40 k tokens) while
  leaving the rest of the GPU for the KServe engine.

### 4.4 RayService spec change = full cluster rotation, not a redeploy

Every edit to the rayService spec creates a NEW pending RayCluster; the new one is
promoted to ACTIVE only when healthy; the old ACTIVE cluster is garbage collected after
promotion. Promoting/stuck states are common. To force-down the old/extra clusters so
rotation can proceed:

```bash
# JSON patch — do NOT use a merge patch here:
kubectl patch rayclusters.ray.io <old-or-pending-cluster> -n default --type=json -p \
  '[{"op":"replace","path":"/spec/workerGroupSpecs/0/replicas","value":0},{"op":"replace","path":"/spec/workerGroupSpecs/0/minReplicas","value":0}]'
```

(A merge patch rewrites the worker group spec and loses the `template` → validation
error.) Old workers must reach 0 before the GC/pruning can finish.

### 4.5 No network / air-gapped images

Everything runs `imagePullPolicy: Never` + locally cached images. Pulling images (or
redownloading the HF model) will fail or hang. Model cache must exist on every node
that hosts the engine; this single-node setup uses `hostPath: /var/lib/ray-hf-cache`.

### 4.6 Reboot behavior

- k3s service is `enabled` → starts on boot; images + `/var/lib/ray-hf-cache` + the
  model persist on disk → **no re-download**.
- After a reboot each engine re-does: torch.compile (~50 s) + CUDA graph capture
  (~60 s). Expect ~2–3 min of `Running ... download model files` / engine warm-up
  before `RUNNING`, per pod. That is normal; do not paper over it.
- If you hit the stuck-download symptom after a reboot while RAM is tight, reduce load
  on the node first (see KServe README §4.4 — same root cause: host RAM pressure).

### 4.7 ServeController zombies / stuck replicas

The Serve controller lives on the head. If references/apps get wedged (e.g. a
"restarting" replica never drains), force a replay from GCS:

```bash
kubectl exec <head-pod> -- bash -c 'kill -9 $(pgrep -f ServeController)'
# ServeController respawns and replays the target application from the GCS store.
```

### 4.8 Don't scale shared GPU blindly

Caution: when the GPU is shared with another engine (KServe), ANY rollout of either
side deadlocks because the replacement pod cannot allocate VRAM while the old pod still
holds it (CUDA OOM, crash-loop, no progress). Free the old ReplicaSet manually before
(or just after) the new pod starts — see KServe README §4.5 for the same trick.

## 5. Troubleshooting cheat-sheet

| Symptom | Cause / fix |
|---|---|
| `No available memory for the cache blocks` | Engine util ≤ 0.25 on 8 GiB; set `0.38` |
| `Task was killed due to the node running low on memory` | memory monitor; add the two env vars (§4.2) |
| App stuck `Running N tasks to download model files on worker nodes...` | worker CPUs; explicit `rayStartParams.num-cpus: 4` (§4.1) |
| New cluster pending forever / old one never GC'd | patch old cluster worker replicas+minReplicas → 0 (§4.4) |
| Engine boots for 2–3 min after reboot | normal; torch.compile + CUDA graphs |
| `404` on `/v1/chat/completions` | use the model id `qwen-0.5b` (not `qwen`) and the worker pod's `8000`; head's OpenAiIngress only serves `/` (§3) |