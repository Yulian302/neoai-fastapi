import numpy as np
from keras.models import load_model
from time import time

from models.source.classifiers.ClModel import ClModel
from models.utils.Image import Image


def init_model(model_name: str) -> any:
    model_name_lower = model_name.lower()
    if model_name_lower == "clmodel":
        return ClModel
    else:
        raise Exception("No models found")


def use_vgg16_model(model_path: str, model_name: str, base64data, image_size):
    try:
        cl_model = init_model(model_name)
        model = load_model(
            model_path,
            custom_objects={model_name: cl_model},
        )
        img_ = Image(data=base64data)
        img_.augment()
        img_.preprocess(required_shape=(image_size, image_size), channels=3)
        start = time()
        predictions = model.predict(img_.image)
        infer_time = time() - start
        predictions = np.argmax(predictions, axis=1)
        prediction_label = get_prediction_label(cl_model, int(predictions[0]))
        return prediction_label, infer_time

    except FileNotFoundError as e:
        raise Exception(f"Model file not found at path: {model_path}") from e

    except Exception as e:
        raise Exception(f"An error using model occurred: {str(e)}") from e


def get_prediction_label(model_class, prediction_value: int) -> str:
    return model_class.predictions[prediction_value]
