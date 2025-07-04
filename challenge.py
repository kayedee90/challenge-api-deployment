from fastapi import FastAPI
from pydantic import BaseModel, model_validator

from typing import Optional

app = FastAPI()

@app.get("/")

def root():
    return {"message": "Hello, this is a GET request!"}

@app.get("/double/{number}")

def double_number(number:int):
    return {"result": number * 2}

@app.post("/hello/")
def say_hello(name: str):
    return {"message": f"Hello, {name}!"}


class SalaryInput(BaseModel):
    salary: float
    bonus: float
    taxes: float

    @model_validator(mode="before")
    def check_fields(cls, values):
        for field in ['salary', 'bonus', 'taxes']:
            value = values.get(field)
            if value is None:
                raise ValueError(f"3 fields expected (salary, bonus, taxes). You forgot: {field}.")
            if not isinstance(value, (int, float)):
                raise ValueError("expected numbers, got strings.")
        return values

@app.post("/calculate/")
def calculate_income(data: SalaryInput):
    result = data.salary + data.bonus - data.taxes
    return {"result": result}
