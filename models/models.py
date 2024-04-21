from pydantic import BaseModel


class ModelData(BaseModel):
    image: str
    name: str
