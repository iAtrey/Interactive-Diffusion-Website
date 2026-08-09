# DiffATS Setup Notes

## Environments needed

**main_env311** — Python 3.11, for inference/reconstruct/render (steps 8-10).
Python 3.14 (system default) has no PyTorch wheels — must use uv to get 3.11.

```bash
pip install uv
uv python install 3.11
uv venv --python 3.11 ~/main_env311
uv pip install --python ~/main_env311/bin/python3.11 -r requirements.txt
```

**Taichi env** — separate Python 3.11 venv, for raw data generation (step 6) only.
```bash
uv venv --python 3.11 ~/taichi_env
uv pip install --python ~/taichi_env/bin/python3.11 torch numpy matplotlib scipy taichi tqdm
```

Both envs are called by full path (e.g. `~/main_env311/bin/python3.11 script.py`) —
never activate them directly, since neither has pip/uv installed inside it.

## Before running anything

- Set `DIFFATS_ROOT` to wherever the shared data/checkpoint lives:
  `export DIFFATS_ROOT=[SHARED_PATH]`
- Set `DIFFATS_CHECKPOINT` to the full checkpoint file path:
  `export DIFFATS_CHECKPOINT=[CHECKPOINT_PATH]`
- Check `generate_data.py`'s hardcoded GPU index matches your actual GPU
  (`nvidia-smi -L` to check, `grep CUDA_VISIBLE_DEVICES generate_data.py` to verify).
  Ignore the "Using GPU (device 4)" print line — it's hardcoded and not the real device.

## Known bugs already patched in this repo — don't re-debug these

1. **`gen_karman_2d.py` broken import**: had `sys.path.insert(0, '${REPO_ROOT}/video')`
   — `${REPO_ROOT}` is shell syntax, doesn't expand in Python. Fixed to point at
   `exps/moving_mnist` directly (shared diffusion-sampling module).

2. **`making_videos_api.py` deprecated matplotlib call**: `mcm.get_cmap()` was
   removed in matplotlib 3.9+. Fixed to `mpl.colormaps.get_cmap()` with
   `import matplotlib as mpl` added.

3. **`making_videos_api.py` hardcoded `SAMPLES_PER_SHARD = 50`**: must match
   your actual shard size (`--clips` value used in step 6). Check with:
```python
   import torch
   shard = torch.load("[path]/shard_000.pt", weights_only=False)
   print(len(shard))
```

4. **`making_videos.py` looks for `test_shard_NNN.pt`**, but `generate_data.py`
   writes `shard_NNN.pt`. Fixed via symlinks:
```bash
   for i in 000 001 002 003; do ln -s shard_${i}.pt test_shard_${i}.pt; done
```
   (adjust the shard numbers to match your actual data)

5. **`gen_karman_2d.py` asserts `test_dataset == 500`** — only true on the full
   10,000-clip run. On smaller test data, either build a real test set or
   remove the assert (lines ~190-191) — but note this if you do, results
   aren't comparable to a full run.

## Running the pipeline

Use `run_generation.py` — takes a dataset name and integer seed, runs
inference → reconstruct → render, returns output paths. See that file directly
for usage.
