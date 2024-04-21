import os
import ssl

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.cors import CORSMiddleware

from models.models import ModelData, MODELS
from models.utils.prediction import use_vgg16_model
from models.utils.segmentation import use_unet_model

app = FastAPI()

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain("./cert.pem", keyfile="./key.pem")

origins = [
    "https://localhost:3000",
    "http://localhost:3000",
    "https://neo-ai-front-rprz-git-main-yulian302s-projects.vercel.app:443",
    "https://neo-ai-front-rprz-git-main-yulian302s-projects.vercel.app",
    "http://neo-ai-front-rprz-git-main-yulian302s-projects.vercel.app",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost",
        "neo-ai-front-rprz-git-main-yulian302s-projects.vercel.app",
    ],
)


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "1.0"}


@app.post("/api/models/{model_id}/use/")
def use_model(payload: ModelData, model_id=1):
    # try:
    format_, img_str = payload.image.split(";base64,")
    model_name = MODELS[model_id]
    print(model_name)
    models_size = {"ClModel": 512, "BrainTumorMriSegmentationUnet": 128}
    prediction_label, infer_time = use_ai_model(
        model_name, img_str, img_size=models_size[model_name]
    )
    return {"prediction_label": prediction_label, "inference_time": infer_time}
    # except Exception as e:
    #     return {"error": "There was an error using a model"}


def use_ai_model(model_name: str, base64data, img_size):
    model_path = os.path.join(
        "/Users/yulianbohomol/PycharmProjects/neoAiModels/models/serialized/",
        f"{model_name.lower()}.h5",
    )
    model_params = {
        "ClModel": lambda: use_vgg16_model(
            model_path, model_name, base64data, img_size
        ),
        "BrainTumorMriSegmentationUnet": lambda: use_unet_model(
            model_path, base64data, img_size
        ),
    }
    pred_, infer_time_ = model_params[model_name]()
    return pred_, infer_time_


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
