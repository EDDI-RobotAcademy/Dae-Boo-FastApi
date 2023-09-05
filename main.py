from fastapi import FastAPI
from app.card_recommend.age_recommend_card import request_receiver

app = FastAPI()

origins = ["http://localhost:3002"]
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root_index():
  return {"message":"Helasdflaksdjf"}

app.include_router(request_receiver)

