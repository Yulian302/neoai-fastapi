from pydantic import BaseModel


class ModelData(BaseModel):
    image: str


MODELS = {"1": "ClModel", "2": "BrainTumorMriSegmentationUnet"}
