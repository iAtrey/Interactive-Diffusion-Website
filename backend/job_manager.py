import time
import uuid
from time import sleep
from threading import Lock, Thread
import threading


"""I kinda forgot pseudocode for this one but basically the first function
 is like all the values needed on startup; the next 3 go into timing it,
 and the code up to mark_fail is the status of the video. """

class LiveUpdate :

    def __init__(self):
        self.current_status = "idle"
        self.current_seed = None
        self.current_dataset = None
        self.current_mode = None
        self.current_sample_idx = None
        self.start = None
        self.end = None
        self.url = None
        self.error = None
        self.job_id = None

        self.lock = threading.Lock()

    def time_start(self) :
        if self.current_status == "running" :
            self.start = time.time()
            return self.start
        else :
            return "no job running"

    def time_end(self) :
        if self.current_status == "finished" :
            self.end = time.time()
            return self.end
        else :
            return "no job running"
    
    def overall_time(self) :
        overall_time = self.end - self.start
        return overall_time


    def start_job(self, seed, dataset, mode=None, sample_idx=None) :
        # returns the new job's id on success, or None if one is already
        # running. callers use the id to tell "my job" apart from whichever
        # job the whiteboard is showing when they poll.
        with self.lock :
            if self.current_status == "running" :
                return None
            else :
                # clear the last job before starting this one, otherwise its
                # end time makes the timer negative and its error text shows
                # up on a run that actually succeeded
                self.end = None
                self.url = None
                self.error = None

                self.current_status = "running"
                self.current_dataset = dataset
                self.current_seed = seed
                self.current_mode = mode
                self.current_sample_idx = sample_idx
                self.job_id = uuid.uuid4().hex
                self.time_start()
                return self.job_id


    def mark_finish(self, video_link) :
        if self.current_status not in ("idle", "failed"):
            with self.lock :
                self.current_status = "finished"
                self.url = video_link
                self.time_end()



    def mark_fail(self, errortext) :
        if self.current_status not in ("idle", "finished"):
            with self.lock :
                self.current_status = "failed"
                self.error = errortext
                # stop the clock, or handout_status keeps counting up
                # every time the frontend polls a job that already died
                self.end = time.time()


    def handout_status (self, job_id=None) :
        current_time = time.time()
        if self.current_status == "idle" :
            return "No job running"
        # a caller polling with an old job's id is asking about a job that's
        # already gone -- don't hand them whatever replaced it
        if job_id is not None and job_id != self.job_id :
            return "unknown job"
        if self.end is not None :
            timer = self.end - self.start
        else:
            timer = current_time - self.start

        return timer, self.current_status, self.error, self.job_id

    def handout_result (self, job_id=None) :
        if job_id is not None and job_id != self.job_id :
            return "unknown job"
        if self.current_status == "finished" and self.url is not None :
            return (self.url, self.current_dataset, self.current_seed,
                    self.job_id, self.current_mode, self.current_sample_idx)
        else :
            return "job not yet done"
        


whiteboard = LiveUpdate()
