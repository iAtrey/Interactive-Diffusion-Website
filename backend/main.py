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

# Setup A/B -> which block of the 100-sample instructor test set to draw
# from. The actual sample id is block_start + seed % 50, so seed both picks
# the diffusion noise and (within the chosen setup) which real sample is shown.
SETUP_BLOCKS = {"A": 0, "B": 50}
BLOCK_SIZE = 50

# keep this many most-recent videos in output/; older ones get deleted once a
# new run finishes successfully
KEEP_VIDEOS = 5

# static file mount -- makes everything in output/ fetchable at /output/...
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
app.mount("/output", StaticFiles(directory=output_dir), name="output")

# serve the UI from here too, so the page and the API share an origin and
# CORS never enters into it
frontend_file = Path(__file__).parent / "frontend" / "index.html"


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


def run_job(inputs: RequiredInputs):
    """Runs in a background thread so /generate can answer immediately
    instead of freezing the whole server for the length of a generation.

    run_generation takes a dataset, a seed, and (now) a sample_idx -- which of
    the 100 real test samples to render. setup A/B picks the block of ids,
    seed picks which id inside that block, so both controls visibly change
    the output. re/niu/cx/cy/r are still accepted for display purposes but
    aren't separately fed to the pipeline -- they come along with whichever
    real sample gets picked.
    """
    sample_idx = SETUP_BLOCKS[inputs.setup] + (inputs.seed % BLOCK_SIZE)
    try:
        paths = run_generation(
            dataset=inputs.dataset, seed=inputs.seed, sample_idx=sample_idx
        )

        # the pipeline writes its mp4 deep inside DIFFATS_ROOT, which the
        # /output mount can't see. Copy it in under a per-seed name so one
        # run can't clobber the last, then hand back the URL form of it.
        source = Path(paths["video_path"])
        slug = inputs.dataset.replace(" ", "_")
        destination = output_dir / f"{slug}_seed{inputs.seed}_id{sample_idx}.mp4"
        shutil.copy2(source, destination)

        _cleanup_intermediates(paths)
        _prune_old_videos()

        whiteboard.mark_finish(f"/output/{destination.name}")
    except Exception as error:
        whiteboard.mark_fail(str(error))


@app.post("/generate", response_model=StatusResponse)
def generate(inputs: RequiredInputs):
    # start_job returns None when a job is already running -- the "one job at a time" rule
    job_id = whiteboard.start_job(inputs.seed, inputs.dataset)
    if job_id is None:
        raise HTTPException(
            status_code=409,
            detail="A job is already running. Check /status and try again.",
        )

    threading.Thread(target=run_job, args=(inputs,), daemon=True).start()
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

    url, dataset, seed, current_id = answer
    return ResultResponse(url=url, dataset=dataset, seed=seed, job_id=current_id)
