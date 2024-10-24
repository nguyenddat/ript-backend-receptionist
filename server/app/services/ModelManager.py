import os
import cv2
import base64
import pickle
import numpy as np
from numpy.linalg import norm
from collections import Counter

from PIL import Image
from typing import Optional, List
from sklearn import neighbors

from insightface.app import FaceAnalysis
    
class ModelManager(object):
    def __init__(self, 
                 save_model_path = "./core/KNNClassifier.pkl", 
                 model = "buffalo_l", 
                 ctx_id = 0, 
                 det_size = (640, 640), 
                 k: int = 3, 
                 similarity_threshold: float = 0.35):
        # Init face embedding model
        self.model = FaceAnalysis(name = model)
        self.model.prepare(ctx_id = ctx_id, det_size = det_size)

        # Initial KNN model parameters
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.save_model_path = save_model_path

        # Storing embeddings, labels and result after training model
        self.knn_clf = None
        self.X_stored, self.y_stored = self._load_stored_data() or ([], [])

        # Storing error
        self.last_error = None
        os.system('cls||clear')

    @staticmethod
    def cosine_similarity(a: np.array, b: np.array):
        norm_a, norm_b = norm(a), norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        try:
            cosine_sim = np.dot(a, b) / (norm(a) * norm(b))
            return cosine_sim
        except:
            return 0
        
    @staticmethod
    def KNN(k, thres_hold, img_array, X_stored, y_stored):
        distances = []
        current_k = min(k, len(y_stored))
        for i in range(len(X_stored)):
            cosine_sim = ModelManager.cosine_similarity(img_array, X_stored[i].embedding)
            if cosine_sim >= thres_hold:
                distances.append([y_stored[i], cosine_sim])
        top_k = sorted(distances, key = lambda x: x[1], reverse = True)[:current_k]
        most_common = Counter([label for label, _ in top_k]).most_common()
        if len(most_common) == 0:
            return "Unknown"
        elif len(most_common) == 1:
            return most_common[0][0]
        else:
            return sorted(top_k, key = lambda x: x[1], reverse = True)[0]
    
    @staticmethod
    def surface(left, upper, right, lower):
        return (right - left) * (lower - upper)
    
    def embed_face(self, img_array):
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        faces = self.model.get(img)

        nums_of_people = len(faces)
        if nums_of_people == 0:
            return [np.zeros((512))], nums_of_people
        return faces, nums_of_people

    def _save_stored_data(self):
        stored_data_path = os.path.splitext(self.save_model_path)[0] + "_store_data.pkl"
        stored_data = {
            "X": self.X_stored,
            "y": self.y_stored
        }
        with open(stored_data_path, "wb") as file:
            pickle.dump(stored_data, file)

    def _load_stored_data(self):
        stored_data_path = os.path.splitext(self.save_model_path)[0] + "_stored_data.pkl"
        if os.path.exists(stored_data_path):
            with open(stored_data_path, "rb") as file:
                stored_data = pickle.load(file)
                return stored_data["X"], stored_data["y"]

    def load_new_data(self, data_dir, imageManager, verbose = True) -> Optional[None]:
        class_dir = data_dir.replace("\\", "/").rstrip("/").split("/")[-1]
        for img in imageManager.image_files_in_folder(data_dir):
                image = imageManager.load_img_file(img)
                faces, nums_of_people = self.embed_face(image)
                if nums_of_people != 1:
                    if verbose:
                        error = f"Not suitbale image for training: {img}"
                        print(error)
                    self.last_error = f"Not suitbale image for training: {img}"
                else:
                    self.X_stored.append(faces[0])
                    self.y_stored.append(class_dir)

        self._save_stored_data()

    def predict(self, img_b64, imageManager):
        result = {}
        others = []
        img_array = imageManager.base64_to_array(img_b64)
        faces, nums_of_people = self.embed_face(img_array)
        HEIGHT, WIDTH = img_array.shape[:2]
        LEFT, RIGHT, UPPER, LOWER = WIDTH // 4, WIDTH // 4 + WIDTH // 2, 0, HEIGHT
 
        if len(faces) == 1:
            face = faces[0].embedding
            person = ModelManager.KNN(self.k, self.similarity_threshold, face, self.X_stored, self.y_stored)
            result.update({"main": person})
        else:
            faces_with_surface = []
            for face in faces:
                person = ModelManager.KNN(self.k, self.similarity_threshold, face.embedding, self.X_stored, self.y_stored)
                left, upper, right, lower = face.bbox.astype(int)
                
                if not ((LEFT <= left < right <= RIGHT) and (UPPER <= upper < lower <= LOWER)):
                    others.append(person)
                    continue
                faces_with_surface.append({"person": person, "surface": ModelManager.surface(left, upper, right, lower)})
                faces_with_surface.sort(key = lambda x: x["surface"])
                for i in range(len(faces_with_surface)):
                    person = faces_with_surface[i]
                    if i == 0:
                        result.update({"main": person["person"]})
                    else:
                        others.append(person["person"])
        result.update({"others": others})
        return result
    
    def get_last_error(self):
        print(self.last_error)