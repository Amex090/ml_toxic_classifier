from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline("sentiment-analysis",   
                      "distilbert-base-uncased-finetuned-sst-2-english")



@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text )[0]
