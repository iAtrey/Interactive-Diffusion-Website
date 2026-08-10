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
    # which real test-sample block to draw from -- A = ids 0-49, B = ids 50-99.
    # the actual id is block_start + seed % 50, computed in main.py.
    setup: Literal["A", "B"]



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
