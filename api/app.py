from fastapi import FastAPI
from fastapi import BackgroundTasks
import uvicorn
app = FastAPI()


@app.get("/health")
def read_root():
    return {"health": "True"}


@app.get("/info")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.post("/train")
async def train_model():
    background_tasks.add_task()
    return {"train": "True"}

if __name__ == "__main__":
    uvicorn.run(app=app)
