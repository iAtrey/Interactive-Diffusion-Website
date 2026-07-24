from pydantic import BaseModel, Field
from typing import Literal





# the required inputs that FastAPI looks at when it makes the generate request.
class RequiredInputs(BaseModel) :

    seed: int = Field( ge = 0, le = 999)
    re: float | None = None
    dataset: Literal[
        "karman vortex street"
    ] 
    niu : float | None = None
    cx: int | None = None
    cy: int | None = None
    r: int | None = None



# The status it can be. I can't like idk the purpose of this file but I think it's for saftey reasons

class StatusResponse (BaseModel) :
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
    framecount : int
    dataset : Literal ["karman vortex street"]
    seed : int = Field( ge=0, le=999)






    
