import os
import ssl

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from models.models import ModelData, MODELS, MODELS_BASE_PATH
from models.utils.prediction import use_vgg16_model
from models.utils.segmentation import use_unet_model

app = FastAPI()

# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# ssl_context.load_cert_chain("/etc/letsencrypt/live/sampledomain.space/fullchain.pem", keyfile="/etc/letsencrypt/live/sampledomain.space/privkey.pem")

origins = ["sampledomain.space","api.sampledomain.space"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["sampledomian.space","api.sampledomain.space","localhost"],
)


@app.get("/")
def home():
    return JSONResponse({"health_check": "OK", "model_version": "1.0"})


@app.post("/api/models/{model_id}/use/")
def use_model(payload: ModelData, model_id: str = "1"):
    try:
        format_, img_str = payload.image.split(";base64,")
        model_info = MODELS[str(model_id)]
        model_name = model_info["name"]
        prediction_label, infer_time = use_ai_model(
            model_name, img_str, img_size=model_info["image_size"]
        )
        return JSONResponse(
            {"prediction_label": prediction_label, "inference_time": infer_time}
        )
    except Exception as e:
        return JSONResponse({"error": e}, status_code=500)


def use_ai_model(model_name: str, base64data, img_size: int):
    model_path = os.path.join(
        MODELS_BASE_PATH,
        f"{model_name}.h5",
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
