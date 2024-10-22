import io
import re
import os
import base64
import numpy as np
from PIL import Image
from typing import Optional

class ImageManager(object):
    def __init__(self):
        self.supported_formats = [".jpg", "jpeg", ".png"]
        self.last_error: str = None

    def load_img_file(self, file_dir, mode = "RGB"):
        img = Image.open(file_dir)
        if mode:
            img = img.convert(mode)
        return np.array(img)

    def image_files_in_folder(self, folder_dir):
        return [os.path.join(folder_dir, f) for f in os.listdir(folder_dir) if re.match(r'.*\.(jpg|jpeg|png)', f, flags = re.I)]

    def img_to_base64(self, file_dir: str) -> Optional[str]:
        if not any(file_dir.lower().endswith(file_ext) for file_ext in [".jpg", "jpeg", ".png"]):
            self.last_error = f"Unsupported format"
            return None
        with open(file_dir, "rb") as file:
                return base64.b64encode(file.read().decode("utf-8"))

    def save_img_from_base64(self, img_b64, file_dir):
        temp = img_b64.split(",")
        if len(temp) != 2:
            self.last_error = f"Invalid base64 string format: {file_dir}"
            return None
        else:
            img_b64 = temp[1]
            img_binary = base64.decode(img_b64)
            image = Image.open(io.BytesIO(img_binary))
            image.save(file_dir)
