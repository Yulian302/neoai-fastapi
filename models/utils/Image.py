import base64
import cv2
import numpy as np
from PIL import ImageEnhance
from PIL import Image as Image_
import random


class Image:
    def __init__(self, data):
        self.base64data = data
        self.image = self.read_data()

    def read_data(self):
        decoded_data = base64.b64decode(self.base64data)
        np_data = np.frombuffer(decoded_data, dtype=np.uint8)
        return cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

    def preprocess(self, required_shape: tuple, channels: int) -> None:
        if self.image.shape != required_shape:
            self.image = cv2.resize(
                self.image, dsize=required_shape, interpolation=cv2.INTER_CUBIC
            )
        self.image = np.reshape(
            self.image, (-1, required_shape[0], required_shape[1], channels)
        )

    def augment(self) -> None:
        self.image = Image_.fromarray(np.uint8((self.image * 255).astype(np.uint8)))
        self.image = ImageEnhance.Brightness(self.image).enhance(
            random.uniform(0.8, 1.2)
        )
        self.image = ImageEnhance.Contrast(self.image).enhance(random.uniform(0.8, 1.2))
        self.image = np.asarray(self.image, dtype=np.float32) / 255.0
