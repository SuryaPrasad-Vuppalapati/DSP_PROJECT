from fastapi import FastAPI


app = FastAPI()

@app.get("/")
def index():
    return {"Data":{"message": "returning from index()"}}

@app.get("/hello")
def hello():
    return {"Data":{"message": "returning from hello()"}}