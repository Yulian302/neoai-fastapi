import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt
from models.source.segmentators.UNetBrainTumorSeg import (
    SEGMENT_CLASSES,
    init_model,
)
from models.utils.Image import Image

VOLUME_SLICES = 100


def predict_single_image(model, image, img_size):
    X = np.empty((VOLUME_SLICES, img_size, img_size, 2))
    ce = image
    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(image[:, :, j + 22], (img_size, img_size))
        X[j, :, :, 1] = cv2.resize(ce[:, :, j + 22], (img_size, img_size))

    return model.predict(X / np.max(X), verbose=1)


def show_predicts_for_single_image(
    model, image, img_size, start_slice=0, save_path="./prediction_results.png"
):
    gt = np.zeros((img_size, img_size))  # Mock ground truth for a single image
    p = predict_single_image(model, image, img_size)

    core = p[:, :, :, 1]
    edema = p[:, :, :, 2]
    enhancing = p[:, :, :, 3]

    plt.figure(figsize=(18, 50))
    f, axarr = plt.subplots(1, 6, figsize=(18, 50))

    for i in range(6):  # for each image, add brain background
        axarr[i].imshow(
            cv2.resize(image[:, :, start_slice + 22], (img_size, img_size)),
            cmap="gray",
            interpolation="none",
        )

    axarr[0].imshow(
        cv2.resize(image[:, :, start_slice + 22], (img_size, img_size)), cmap="gray"
    )
    axarr[0].title.set_text("Original image")
    curr_gt = cv2.resize(gt, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    axarr[2].imshow(
        p[start_slice, :, :, 1:4], cmap="Reds", interpolation="none", alpha=0.3
    )
    axarr[2].title.set_text("All classes")
    axarr[3].imshow(
        edema[start_slice, :, :], cmap="OrRd", interpolation="none", alpha=0.3
    )
    axarr[3].title.set_text(f"{SEGMENT_CLASSES[1]} predicted")
    axarr[4].imshow(
        core[start_slice, :, :], cmap="OrRd", interpolation="none", alpha=0.3
    )
    axarr[4].title.set_text(f"{SEGMENT_CLASSES[2]} predicted")
    axarr[5].imshow(
        enhancing[start_slice, :, :], cmap="OrRd", interpolation="none", alpha=0.3
    )
    axarr[5].title.set_text(f"{SEGMENT_CLASSES[3]} predicted")

    plt.savefig(save_path)


def use_unet_model(model_path: str, base64data, img_size):
    try:
        model = init_model(img_size)
        start = time()
        img_ = Image(data=base64data)
        img_.preprocess(required_shape=(img_size, img_size), channels=1)
        predictions = "pred"
        show_predicts_for_single_image(model, img_.image, img_size)
        infer_time = time() - start
        predictions = np.argmax(predictions, axis=1)
        print(predictions)
        # prediction_label = get_prediction_label(cl_model, int(predictions[0]))
        prediction_label = "good"
        return prediction_label, infer_time

    except FileNotFoundError as e:
        raise Exception(f"Model file not found at path: {model_path}") from e

    except Exception as e:
        raise Exception(f"An error using model occurred: {str(e)}") from e


#
# use_unet_model(
#     "../serialized/braintumormrisegmentationunet.h5",
#     "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",
#     img_size=128,
# )
