import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from schemas import RequiredInputs, StatusResponse, ResultResponse
from job_manager import whiteboard
from run_generation import run_generation

app = FastAPI()

# allowed hosts -- "*" accepts any Host header; fix once there's a real domain
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# CORS -- only the tunneled page itself is allowed to call this API. The
# browser is reaching this server through `ssh -L 8000:localhost:8000`, so
# the page's own origin is always exactly this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Which block of samples each setup maps to, per mode: setup -> (start, size).
# The sample actually rendered is start + seed % size, so the setup picks the
# parameter group and the seed picks a specific sample inside it.
#
# The two modes have genuinely different data, so their setups differ:
#   quick -- 15 clips in 3 groups of 5   (A: Re 70, B: Re 90, C: Re 70 shifted)
#   full  -- 100 clips in 2 groups of 50 (A: Re 76.19, B: Re 69.57)
# Setup C exists only in quick mode; asking for it in full mode is rejected.
MODE_SETUPS = {
    "quick": {"A": (0, 5), "B": (5, 5), "C": (10, 5)},
    "full":  {"A": (0, 50), "B": (50, 50)},
}

# keep this many most-recent videos in output/; older ones get deleted once a
# new run finishes successfully
KEEP_VIDEOS = 5

# static file mount -- makes everything in output/ fetchable at /output/...
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
app.mount("/output", StaticFiles(directory=output_dir), name="output")

# serve the UI from here too, so the page and the API share an origin and
# CORS never enters into it
frontend_file = Path(__file__).parent / "index.html"


@app.get("/")
def home():
    if not frontend_file.exists():
        raise HTTPException(status_code=404, detail=f"UI not found at {frontend_file}")
    return FileResponse(frontend_file)


def _prune_old_videos():
    """Keep only the KEEP_VIDEOS most recently finished videos in output/."""
    videos = sorted(
        output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for stale in videos[KEEP_VIDEOS:]:
        stale.unlink(missing_ok=True)


def _cleanup_intermediates(paths: dict):
    """Delete the big working files a run leaves behind once its mp4 is
    safely copied into output/. factors_path + reconstructed_path are the
    real disk hogs (~0.18 GB per run); the per-run render folder holds gifs
    that are never served."""
    for key in ("factors_path", "reconstructed_path"):
        p = Path(paths[key])
        p.unlink(missing_ok=True)
    render_dir = Path(paths["video_path"]).parent
    shutil.rmtree(render_dir, ignore_errors=True)


def resolve_sample_idx(mode: str, setup: str, seed: int) -> int:
    """Turn a mode + setup + seed into one specific test-sample id.

    Raises KeyError if the setup doesn't exist in that mode (e.g. C in full).
    """
    start, size = MODE_SETUPS[mode][setup]
    return start + (seed % size)


def run_job(inputs: RequiredInputs, sample_idx: int):
    """Runs in a background thread so /generate can answer immediately
    instead of freezing the whole server for the length of a generation.

    sample_idx is resolved by the caller (so a bad setup/mode pairing gets
    rejected before a job is ever started). re/niu/cx/cy/r are accepted for
    display purposes but aren't separately fed to the pipeline -- they come
    along with whichever real sample gets picked.
    """
    try:
        paths = run_generation(
            dataset=inputs.dataset,
            seed=inputs.seed,
            sample_idx=sample_idx,
            mode=inputs.mode,
        )

        # the pipeline writes its mp4 deep inside DIFFATS_ROOT, which the
        # /output mount can't see. Copy it in under a name carrying mode +
        # seed + sample id, so no two runs can clobber each other.
        source = Path(paths["video_path"])
        slug = inputs.dataset.replace(" ", "_")
        destination = (
            output_dir / f"{slug}_{inputs.mode}_seed{inputs.seed}_id{sample_idx}.mp4"
        )
        shutil.copy2(source, destination)

        _cleanup_intermediates(paths)
        _prune_old_videos()

        whiteboard.mark_finish(f"/output/{destination.name}")
    except Exception as error:
        whiteboard.mark_fail(str(error))


@app.post("/generate", response_model=StatusResponse)
def generate(inputs: RequiredInputs):
    # Reject an impossible setup/mode pairing before touching the GPU.
    # Setup C only exists in quick mode, where the data has three groups.
    try:
        sample_idx = resolve_sample_idx(inputs.mode, inputs.setup, inputs.seed)
    except KeyError:
        valid = ", ".join(sorted(MODE_SETUPS[inputs.mode]))
        raise HTTPException(
            status_code=422,
            detail=(
                f"Setup {inputs.setup} does not exist in {inputs.mode} mode. "
                f"Valid setups: {valid}."
            ),
        )

    # start_job returns None when a job is already running -- the "one job at a time" rule
    job_id = whiteboard.start_job(
        inputs.seed, inputs.dataset, mode=inputs.mode, sample_idx=sample_idx
    )
    if job_id is None:
        raise HTTPException(
            status_code=409,
            detail="A job is already running. Check /status and try again.",
        )

    threading.Thread(
        target=run_job, args=(inputs, sample_idx), daemon=True
    ).start()
    return status(job_id=job_id)


@app.get("/status", response_model=StatusResponse)
def status(job_id: str | None = None):
    answer = whiteboard.handout_status(job_id)

    # handout_status returns a bare string when there's nothing to report --
    # either no job has ever run, or job_id names one that's no longer current.
    if isinstance(answer, str):
        if answer == "unknown job":
            raise HTTPException(status_code=404, detail="No job with that id.")
        return StatusResponse(job_id=None, time=None, state="idle", errormessage=None)

    elapsed, state, error_message, current_id = answer
    return StatusResponse(
        job_id=current_id, time=elapsed, state=state, errormessage=error_message
    )


@app.get("/result", response_model=ResultResponse)
def result(job_id: str | None = None):
    answer = whiteboard.handout_result(job_id)

    if isinstance(answer, str):
        if answer == "unknown job":
            raise HTTPException(status_code=404, detail="No job with that id.")
        raise HTTPException(
            status_code=404,
            detail="No finished video yet. Check /status to see what's happening.",
        )

    url, dataset, seed, current_id, mode, sample_idx = answer
    return ResultResponse(
        url=url, dataset=dataset, seed=seed, job_id=current_id,
        mode=mode, sample_idx=sample_idx,
    )
