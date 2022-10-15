from typing import Optional
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import os
import base64

from Dector import Detector


class PictureReq(BaseModel):
    pic_base64: Optional[str]
    pic_url: Optional[str]
    accuracy: Optional[float] = 0.8


def cache_file(b: bytes) -> str:
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    rand = time.time_ns()
    with open(f"tmp/{rand}.jpg", "wb") as binary_file:
        binary_file.write(b)
    return f"tmp/{rand}.jpg"


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
    if req.pic_base64:
        b = base64.b64decode(req.pic_base64)
        path = cache_file(b)
    elif req.pic_url:
        async with httpx.AsyncClient() as client:
            r = await client.get(req.pic_url)
            path = cache_file(r.content)
    else:
        raise HTTPException(status_code=400, detail="No pic")
    try:
        loong = d.detect(path, req.accuracy)
        os.remove(path)
        return {"loong": f"{loong}"}
    except Exception as e:
        os.remove(path)
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
