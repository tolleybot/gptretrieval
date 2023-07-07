import uvicorn
from fastapi import FastAPI, HTTPException, Body

from models.api import QueryRequest, QueryResponse
from datastore.factory import get_datastore
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

PORT = 3333

origins = [
    f"http://localhost:{PORT}",
    "https://chat.openai.com",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.route("/.well-known/ai-plugin.json")
async def get_logo(request):
    file_path = "./data/ai-plugin.json"
    return FileResponse(file_path, media_type="text/json")


@app.route("/logo.png")
async def get_logo(request):
    file_path = "./data/logo.png"
    return FileResponse(file_path, media_type="image/png")


@app.route("/openapi.yaml")
async def get_openapi(request):
    file_path = "./data/openapi.yaml"
    return FileResponse(file_path, media_type="text/json")


@app.post("/query", response_model=QueryResponse)
async def query_main(request: QueryRequest = Body(...)):
    try:
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.on_event("startup")
async def startup():
    global datastore
    datastore = await get_datastore()


def start():
    uvicorn.run("server:app", host="localhost", port=PORT, reload=True)


# create main entry point
if __name__ == "__main__":
    start()
