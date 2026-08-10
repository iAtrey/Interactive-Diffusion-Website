# DiffATS Setup Notes

These notes describe how to run the DiffATS Karman-vortex generation
pipeline and its web API on **any** GPU machine ; nothing here is specific
to one server, one username, or one home directory. Everywhere a path is
needed, it comes from an environment variable with a sensible default, not a
hardcoded value.

## Environment variables

Set these before running anything. If unset, the defaults shown are used.

| Variable | Default | What it is |
|---|---|---|
| `DIFFATS_ROOT` | `~/DiffATS` | Path to the cloned DiffATS repo |
| `DIFFATS_CHECKPOINT` | `/shared/checkpoints/karman_vortex_2d_epoch00500_step0156000.pt` | The pretrained Karman-vortex checkpoint |
| `DIFFATS_PYTHON` | `~/diffats_env/bin/python3` | A Python interpreter with torch + CUDA working |
| `DIFFATS_BACKEND` | `~/backend` | Where you've placed `main.py`, `run_generation.py`, etc. |
| `DIFFATS_RENDERER` | `$DIFFATS_BACKEND/making_videos_api.py` | Path to the renderer script. Only needs setting if it doesn't live inside `DIFFATS_BACKEND`. |

Example for a machine where the repo, checkpoint, environment, backend code,
and renderer all live somewhere other than the defaults:

```bash
export DIFFATS_ROOT=/data/DiffATS
export DIFFATS_CHECKPOINT=/data/checkpoints/karman_vortex_2d_epoch00500_step0156000.pt
export DIFFATS_PYTHON=/opt/envs/diffats/bin/python3
export DIFFATS_BACKEND=/srv/diffats-backend
export DIFFATS_RENDERER=/srv/diffats-rendering/making_videos_api.py
```

## Getting a working Python environment

You need one Python environment with torch + CUDA working, and it must be
reachable by an absolute path (never activated implicitly ; every stage is
launched as `<that path> script.py ...`, not `python script.py`).

If your system Python already has a working `torch` with
`torch.cuda.is_available() == True`, use it directly:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If not, build a private one ; the version of Python matters less than
whether a torch wheel exists for it, so check before you build:

```bash
pip install uv
uv python install 3.11
uv venv --python 3.11 ~/diffats_env
uv pip install --python ~/diffats_env/bin/python3.11 torch numpy matplotlib pillow
```

`requirements-pipeline.txt` in this repo pins exact working versions
(including a CUDA-specific torch build) if you want a reproducible install
rather than "whatever's newest" ; see the comments at the top of that file
for the extra `--index-url` it needs, since CUDA-tagged torch builds aren't
on plain PyPI.

A **separate** environment is needed only if you're generating raw
simulation data from scratch (`generate_data.py`, `save_tucker_karman.py`),
since that requires `taichi`, which may not have a wheel for your system
Python version. That environment is unrelated to serving videos ; the API
never touches it.

## Before running anything

- Confirm a GPU is visible: `nvidia-smi -L`.
- If you're regenerating raw data, check `generate_data.py`'s hardcoded GPU
  index matches: `grep CUDA_VISIBLE_DEVICES generate_data.py`. Ignore its
  "Using GPU (device 4)" print line ; that's a hardcoded label, not the real
  device.
- Confirm `ffmpeg` is on `PATH` (`which ffmpeg`) ; the renderer needs it and
  fails with a clear message if it's missing.

## Which data the pipeline uses

The web app generates from the **100-sample test set** shipped alongside the
checkpoint (referred to here as `instructor_test`, since that's this
project's folder name for it ; rename freely if yours is called something
else, see "Adapting to a different layout" below).

That set has two real parameter setups, 50 samples each:

- **Setup A** (ids 0–49): niu=0.021, Re=76.19, cx=27, cy=58, r=8
- **Setup B** (ids 50–99): niu=0.023, Re=69.57, cx=30, cy=64, r=8

These aren't freely editable ; the model can only condition on a real
stored sample, so the UI's setup choice is the entire space of valid input.
If your test set has different parameters or a different sample count, see
below for what to change.

## Adapting to a different data layout

Everything above (the five env vars) is fully general. What isn't
environment-driven is the **internal folder structure** under
`DIFFATS_ROOT` ; that's a property of how the DiffATS repo itself is laid
out, not of the machine. If your checkout uses different folder names, edit
the `DATASETS["karman vortex street"]` dict at the top of
`run_generation.py`:

```python
DATASETS = {
    "karman vortex street": {
        "tucker_dir": f"{DIFFATS_ROOT}/.../data_tucker/instructor_test",
        "gt_dir":     f"{DIFFATS_ROOT}/.../generate/output_instructor_test/gt_shards",
        ...
    }
}
```

If your test set has a different sample count or a different number of
setups than 100/2×50, also update:

- `SAMPLES_PER_SHARD` in `making_videos_api.py` (must match your shard size)
- `SETUP_BLOCKS` and `BLOCK_SIZE` in `main.py` (which block of ids each
  setup maps to)

## Known bugs already patched ; don't re-debug these

1. **`gen_karman_2d.py` broken import**: had `sys.path.insert(0,
   '${REPO_ROOT}/video')` ; `${REPO_ROOT}` is shell syntax, doesn't expand
   in Python. Fixed to point at `exps/moving_mnist` directly.
2. **`making_videos_api.py` deprecated matplotlib call**: `mcm.get_cmap()`
   was removed in matplotlib 3.9+. Fixed to `mpl.colormaps.get_cmap()` with
   `import matplotlib as mpl` added.
3. **`making_videos_api.py` hardcoded `SAMPLES_PER_SHARD`**: must match
   whichever test set you're pointed at. Currently `50`, matching the
   100-sample set described above.
4. **`gen_karman_2d.py` asserts `test_dataset == 500`**: only true on a full
   10,000-clip run. Removed (originally lines ~190–191) since the test set
   in use has 100 samples. If you restore the original file, re-remove it.
5. **Ground-truth shard naming**: the renderer looks for files named
   `test_shard_NNN.pt`. If your raw data was generated with a different
   naming scheme (e.g. plain `shard_NNN.pt`), symlink or rename to match
   before pointing `gt_dir` at it.

## Running the pipeline

`run_generation.py` orchestrates all three stages ; inference, reconstruct,
render ; behind one function:

```python
run_generation(dataset: str, seed: int, sample_idx: int | None = None) -> dict
```

- `dataset` ; must match a key in `DATASETS` (currently only
  `"karman vortex street"` is defined).
- `seed` ; integer, 0–999. Controls the diffusion sampling noise.
- `sample_idx` ; integer, 0–99. Which real test sample to render. The
  mapping from a UI-facing "setup" choice to a specific `sample_idx` lives
  in `main.py` (`SETUP_BLOCKS`), not in `run_generation.py` ; that function
  just takes the final integer. Leaving `sample_idx` unset falls back to the
  renderer's own fixed pick.

One job runs at a time on the GPU ; a second call while one is in flight
raises `BusyError` rather than queuing, enforced with a real file lock (safe
across separate processes, not just separate threads). Every run logs to
`$DIFFATS_BACKEND/logs/`, and a failed stage's real traceback is captured in
the raised `StageError` rather than just an exit code.

For quick manual testing from the command line:

```bash
$DIFFATS_PYTHON run_generation.py <seed>
```

(this omits `sample_idx`, so it renders the renderer's default pick rather
than a specific setup.)

## Serving it

`main.py` (FastAPI) is what a web UI talks to. It:

- computes `sample_idx` from the setup choice + seed and calls
  `run_generation`
- copies the finished mp4 into `$DIFFATS_BACKEND/output/`, deletes the large
  intermediate `.pt` files, and prunes old videos down to the most recent
  `KEEP_VIDEOS` (a constant in `main.py`, defaults to 5)
- issues a job id per run so `/status` and `/result` can't be confused by a
  stale poll from an old request
- serves the frontend itself at `/`, so the browser and the API share one
  origin ; no cross-origin requests, no CORS complexity. It expects
  `index.html` sitting directly in `$DIFFATS_BACKEND` (next to `main.py`) ;
  no subfolder.

Start it with:

```bash
cd "$DIFFATS_BACKEND"
"$DIFFATS_PYTHON" -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Binding to `127.0.0.1` (not `0.0.0.0`) means it's unreachable except through
an SSH tunnel to this machine, even if a network firewall is otherwise open:

```bash
ssh -N -L 8000:localhost:8000 <user>@<host>
```

Then open `http://localhost:8000` in a browser on your own machine.

If you deploy this somewhere with a real domain and multiple simultaneous
users, two things in `main.py` need revisiting: `allow_origins` in the CORS
middleware currently only permits `http://localhost:8000`, and the job
tracker (`job_manager.py`) holds exactly one job's state globally ; fine for
one person behind one tunnel, not for concurrent users.
