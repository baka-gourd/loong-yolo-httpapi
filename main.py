from typing import Optional

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import time
import os
import base64

from Dector import Detector


class PictureReq(BaseModel):
    pic: str
    accuracy: Optional[float] = 0.8


def cache_file(b: bytes) -> str:
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    rand = time.time_ns()
    with open(f'tmp/{rand}.jpg', 'wb') as binary_file:
        binary_file.write(b)
    return f'tmp/{rand}.jpg'


d = Detector()
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/pics/")
async def detect_pic(req: PictureReq):
    tmp = base64.b64decode(req.pic)
    path = cache_file(tmp)
    loong = d.detect(path, req.accuracy)
    # loong = d.detect(str, req.accuracy)
    os.remove(path)
    return {"loong": f'{loong}'}
