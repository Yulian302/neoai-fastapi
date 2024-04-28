from pydantic import BaseModel
from typing import TypedDict


class ModelData(BaseModel):
    image: str


class ModelInfo(TypedDict):
    name: str
    image_size: int


MODELS: dict[str, ModelInfo] = {
    "1": {"name": "ClModel", "image_size": 512},
    "2": {"name": "BrainTumorMriSegmentationUnet", "image_size": 128},
}

MODELS_BASE_PATH = "/Users/yulianbohomol/PycharmProjects/neoAiModels/models/serialized/"
