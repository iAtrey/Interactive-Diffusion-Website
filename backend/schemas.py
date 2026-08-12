from pydantic import BaseModel, Field
from typing import Literal



#I lowk coded ts by myself no AI bc it kept messing it up; I lowk did that for all 3 of the files.


# the required inputs that FastAPI looks at when it makes the generate request.
class RequiredInputs(BaseModel) :

    seed: int = Field( ge = 1, le = 999)
    re: float | None = None
    dataset: Literal[
        "karman vortex street"
    ]
    niu : float | None = None
    cx: int | None = None
    cy: int | None = None
    r: int | None = None
    # which test set to run against.
    #   quick = 15 clips,  ~3 min,  3 setups of 5   (A, B, C)
    #   full  = 100 clips, ~20 min, 2 setups of 50  (A, B)
    mode: Literal["quick", "full"]
    # which block of samples to draw from within that mode. Which letters are
    # valid depends on the mode, so main.py does that check -- C is only real
    # in quick mode. The actual sample id is block_start + seed % block_size.
    setup: Literal["A", "B", "C"]



# The status it can be.

class StatusResponse (BaseModel) :
    job_id: str | None = None
    time: float | None = None
    state : Literal [
        "idle",
        "running",
        "finished",
        "failed"
    ]

    errormessage: str | None = None


class ResultResponse (BaseModel) :
    url : str
    dataset : Literal ["karman vortex street"]
    seed : int = Field( ge=1, le=999)
    job_id: str | None = None
    mode: Literal["quick", "full"] | None = None
    sample_idx: int | None = None
