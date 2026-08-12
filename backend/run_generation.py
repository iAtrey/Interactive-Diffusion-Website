import fcntl
import logging
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DIFFATS_ROOT = os.environ.get("DIFFATS_ROOT", os.path.expanduser("~/DiffATS"))
CHECKPOINT_PATH = os.environ.get(
    "DIFFATS_CHECKPOINT",
    "/shared/checkpoints/karman_vortex_2d_epoch00500_step0156000.pt",
)

BACKEND_DIR = os.environ.get("DIFFATS_BACKEND", os.path.expanduser("~/backend"))

# making_videos_api.py doesn't have to live inside DIFFATS_BACKEND -- point
# this at wherever it actually is. Defaults to BACKEND_DIR for anyone who
# hasn't moved it.
RENDERER_SCRIPT = os.environ.get(
    "DIFFATS_RENDERER", os.path.join(BACKEND_DIR, "making_videos_api.py")
)

_KARMAN = f"{DIFFATS_ROOT}/exps/tensor_physics/exp_karman_vortex"

DATASETS = {
    "karman vortex street": {
        "generate_dir": f"{_KARMAN}/generate",
        "tools_dir": f"{DIFFATS_ROOT}/exps/tensor_physics/tools",
        "checkpoint": CHECKPOINT_PATH,
        "epoch_tag": "epoch00500",
        "epoch": 500,
        "batch_size": 50,
        "sample_steps": 250,
        "device": "cuda:0",
        # Two test sets to choose between. Everything that differs between a
        # fast run and a full-quality run lives here.
        "modes": {
            # 15 clips we generated ourselves. Three parameter groups of 5.
            # Ground truth comes straight from the raw simulation shards,
            # which already carry vor + all params (symlinked to the
            # test_shard_NNN.pt names the renderer expects).
            "quick": {
                "tucker_dir": f"{_KARMAN}/data_tucker/tucker_test",
                "gt_dir": f"{_KARMAN}/data_generation/data_test",
                "n_samples": 15,
                "samples_per_shard": 5,
                "approx_minutes": 3,
            },
            # The instructor's 100-sample set: two parameter setups of 50.
            # Anchored to the checkpoint's own reference, so output quality is
            # noticeably better than quick mode.
            "full": {
                "tucker_dir": f"{_KARMAN}/data_tucker/instructor_test",
                "gt_dir": f"{_KARMAN}/generate/output_instructor_test/gt_shards",
                "n_samples": 100,
                "samples_per_shard": 50,
                "approx_minutes": 20,
            },
        },
    }
}

DEFAULT_MODE = "full"

MAIN_PY = os.environ.get(
    "DIFFATS_PYTHON", os.path.expanduser("~/diffats_env/bin/python3")
)

LOG_DIR = Path(BACKEND_DIR) / "logs"
LOCK_PATH = Path(BACKEND_DIR) / ".gpu.lock"


class BusyError(RuntimeError):
    pass


class StageError(RuntimeError):
    def __init__(self, stage, returncode, log_path, tail):
        self.stage = stage
        self.returncode = returncode
        self.log_path = str(log_path)
        self.tail = tail
        super().__init__(
            f"stage '{stage}' failed with exit code {returncode}. "
            f"Full log: {log_path}\n--- last lines of {stage} ---\n{tail}"
        )


LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("diffats.run_generation")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    _fh = logging.FileHandler(LOG_DIR / "run_generation.log")
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    logger.addHandler(_sh)


def _tail(path, n=40):
    try:
        with open(path, "r", errors="replace") as fh:
            return "".join(fh.readlines()[-n:]).rstrip()
    except OSError as exc:
        return f"(could not read {path}: {exc})"


@contextmanager
def _single_job():
    handle = open(LOCK_PATH, "w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            raise BusyError(
                f"a generation job is already running (GPU lock held: {LOCK_PATH})"
            )
        handle.write(f"pid={os.getpid()} started={datetime.now().isoformat()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def _run_stage(name, argv, cwd, log_path):
    logger.info("stage %-11s starting", name)
    started = time.time()

    with open(log_path, "a") as log_file:
        log_file.write(f"\n===== stage {name} @ {datetime.now().isoformat()} =====\n")
        log_file.write("$ " + " ".join(argv) + "\n")
        log_file.flush()
        proc = subprocess.run(argv, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT)

    elapsed = time.time() - started

    if proc.returncode != 0:
        tail = _tail(log_path)
        logger.error(
            "stage %-11s FAILED after %.1fs (exit %d)\n%s",
            name, elapsed, proc.returncode, tail,
        )
        raise StageError(name, proc.returncode, log_path, tail)

    logger.info("stage %-11s ok in %.1fs", name, elapsed)
    return elapsed


def run_generation(
    dataset: str,
    seed: int,
    sample_idx: int | None = None,
    mode: str = DEFAULT_MODE,
) -> dict:
    """Generate one Karman-vortex comparison video.

    mode picks which test set to run against -- "quick" (15 clips, ~3 min) or
    "full" (100 clips, ~20 min, better quality). It decides the data dirs, how
    many samples exist, and how the ground-truth shards are sliced.

    sample_idx picks which real test sample gets rendered. Its valid range
    depends on the mode (0-14 for quick, 0-99 for full). Left unset, the
    renderer falls back to its own fixed pick.

    seed controls the diffusion sampling noise, independent of which sample is
    shown."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    cfg = DATASETS[dataset]

    if mode not in cfg["modes"]:
        raise ValueError(
            f"Unknown mode: {mode!r}. Expected one of {sorted(cfg['modes'])}"
        )
    mcfg = cfg["modes"][mode]

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Seed must be an integer, got {type(seed).__name__}: {seed!r}")
    if not (0 <= seed <= 999):
        raise ValueError(f"Seed out of range [0, 999]: {seed}")
    if sample_idx is not None:
        if isinstance(sample_idx, bool) or not isinstance(sample_idx, int):
            raise ValueError(f"sample_idx must be an integer: {sample_idx!r}")
        last = mcfg["n_samples"] - 1
        if not (0 <= sample_idx <= last):
            raise ValueError(
                f"sample_idx out of range [0, {last}] for mode {mode!r}: {sample_idx}"
            )

    # One output dir per mode. The two modes produce identically-named
    # intermediates (epoch00500_seed{N}.pt), so sharing a directory would let a
    # stale 15-sample file get picked up by a 100-sample run.
    output_dir = Path(cfg["generate_dir"]) / "output" / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"{mode}_seed{seed}_{stamp}.log"
    timings = {}

    with _single_job():
        job_started = time.time()
        logger.info(
            "job start   dataset=%r mode=%r seed=%d sample_idx=%s "
            "steps=%d batch=%d (~%d min) log=%s",
            dataset, mode, seed, sample_idx, cfg["sample_steps"],
            cfg["batch_size"], mcfg["approx_minutes"], log_path,
        )

        try:
            # inference
            timings["inference"] = _run_stage(
                "inference",
                [
                    MAIN_PY, "gen_karman_2d.py",
                    "--ckpt", cfg["checkpoint"],
                    "--output_dir", str(output_dir),
                    "--train_data_dir", mcfg["tucker_dir"],
                    "--test_data_dir", mcfg["tucker_dir"],
                    "--batch_size", str(cfg["batch_size"]),
                    "--seeds", str(seed),
                    "--sample_steps", str(cfg["sample_steps"]),
                    "--epoch_tag", cfg["epoch_tag"],
                    "--device", cfg["device"],
                ],
                cwd=cfg["generate_dir"],
                log_path=log_path,
            )
            factors_path = output_dir / f"{cfg['epoch_tag']}_seed{seed}.pt"

            # reconstruct
            timings["reconstruct"] = _run_stage(
                "reconstruct",
                [
                    MAIN_PY, "reconstruct_gen.py",
                    "--exp", "karman",
                    "--seed", str(seed),
                    "--epoch", str(cfg["epoch"]),
                    "--dir", str(output_dir),
                ],
                cwd=cfg["tools_dir"],
                log_path=log_path,
            )
            videos_path = output_dir / f"{cfg['epoch_tag']}_seed{seed}_videos.pt"

            # render
            tag = f"seed{seed}" if sample_idx is None else f"seed{seed}_id{sample_idx}"
            video_out_dir = output_dir / "videos" / tag
            render_argv = [
                MAIN_PY, RENDERER_SCRIPT,
                "--gen_path", str(videos_path),
                "--gt_dir", mcfg["gt_dir"],
                "--out_dir", str(video_out_dir),
                "--samples_per_shard", str(mcfg["samples_per_shard"]),
            ]
            if sample_idx is not None:
                render_argv += ["--sample_idx", str(sample_idx)]
            timings["render"] = _run_stage(
                "render", render_argv,
                cwd=os.path.dirname(RENDERER_SCRIPT) or ".",
                log_path=log_path,
            )

            result_video = video_out_dir / "karman_vortex_gt_vs_gen_redblue.mp4"
            if not result_video.exists():
                raise RuntimeError(
                    f"Pipeline finished but expected output is missing: {result_video}"
                )

        except Exception as exc:
            logger.exception(
                "job FAILED   dataset=%r mode=%r seed=%d after %.1fs: %s",
                dataset, mode, seed, time.time() - job_started, exc,
            )
            raise

        timings["total"] = time.time() - job_started
        logger.info(
            "job ok       dataset=%r mode=%r seed=%d in %.1fs "
            "(inference %.1fs, reconstruct %.1fs, render %.1fs) -> %s",
            dataset, mode, seed, timings["total"],
            timings["inference"], timings["reconstruct"], timings["render"],
            result_video,
        )

    return {
        "dataset": dataset,
        "mode": mode,
        "seed": seed,
        "sample_idx": sample_idx,
        "factors_path": str(factors_path),
        "reconstructed_path": str(videos_path),
        "video_path": str(result_video),
        "timings": timings,
        "log_path": str(log_path),
    }


if __name__ == "__main__":
    import sys

    # usage: run_generation.py [seed] [mode] [sample_idx]
    seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODE
    idx_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None
    print(run_generation("karman vortex street", seed_arg,
                         sample_idx=idx_arg, mode=mode_arg))
