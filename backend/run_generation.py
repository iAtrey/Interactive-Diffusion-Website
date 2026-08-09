import subprocess
import os
from pathlib import Path

DIFFATS_ROOT = os.environ.get("DIFFATS_ROOT", os.path.expanduser("~/DiffATS"))
CHECKPOINT_PATH = os.environ.get("DIFFATS_CHECKPOINT", "")

DATASETS = {
    "karman vortex street": {
        "generate_dir": f"{DIFFATS_ROOT}/exps/tensor_physics/exp_karman_vortex/generate",
        "tucker_dir": f"{DIFFATS_ROOT}/exps/tensor_physics/exp_karman_vortex/data_tucker/tucker_dir",
        "tools_dir": f"{DIFFATS_ROOT}/exps/tensor_physics/tools",
        "viz_dir": f"{DIFFATS_ROOT}/exps/tensor_physics/visualization",
        "gt_dir": f"{DIFFATS_ROOT}/exps/tensor_physics/exp_karman_vortex/data_generation/tucker_ouput",
        "checkpoint": CHECKPOINT_PATH,
        "epoch_tag": "epoch00500",
        "epoch": 500,
    }
}

MAIN_PY = os.path.expanduser("~/main_env311/bin/python3.11")

def run_generation(dataset: str, seed: int) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not isinstance(seed, int) or not (0 <= seed <= 999):
        raise ValueError(f"Seed out of range: {seed}")

    cfg = DATASETS[dataset]
    output_dir = Path(cfg["generate_dir"]) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # inference
    subprocess.run(
        [
            MAIN_PY, "gen_karman_2d.py",
            "--ckpt", cfg["checkpoint"],
            "--output_dir", str(output_dir),
            "--train_data_dir", cfg["tucker_dir"],
            "--test_data_dir", f"{cfg['tucker_dir']}/test_data",
            "--batch_size", "50",
            "--seeds", str(seed),
            "--sample_steps", "250",
            "--epoch_tag", cfg["epoch_tag"],
            "--device", "cuda:0",
        ],
        cwd=cfg["generate_dir"], check=True,
    )
    factors_path = output_dir / f"{cfg['epoch_tag']}_seed{seed}.pt"

    # reconstruct
    subprocess.run(
        [
            MAIN_PY, "reconstruct_gen.py",
            "--exp", "karman",
            "--seed", str(seed),
            "--epoch", str(cfg["epoch"]),
            "--dir", str(output_dir),
        ],
        cwd=cfg["tools_dir"], check=True,
    )
    videos_path = output_dir / f"{cfg['epoch_tag']}_seed{seed}_videos.pt"

    # render
    video_out_dir = output_dir / "videos"
    subprocess.run(
        [
            MAIN_PY, "making_videos_api.py",
            "--gen_path", str(videos_path),
            "--gt_dir", cfg["gt_dir"],
            "--out_dir", str(video_out_dir),
        ],
        cwd=cfg["viz_dir"], check=True,
    )

    result_video = video_out_dir / "karman_vortex_gt_vs_gen_redblue.mp4"
    if not result_video.exists():
        raise RuntimeError("Pipeline finished but expected output was not found")

    return {
        "factors_path": str(factors_path),
        "reconstructed_path": str(videos_path),
        "video_path": str(result_video),
    }

if __name__ == "__main__":
    result = run_generation("karman vortex street", 0)
    print(result)
