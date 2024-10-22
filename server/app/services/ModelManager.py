import os
import cv2
import base64
import pickle
from PIL import Image
from typing import Optional
from sklearn import neighbors
from insightface.app import FaceAnalysis
    
class ModelManager(object):
    def __init__(self, 
                 save_model_path: str = None, 
                 model = "buffalo_l", 
                 ctx_id = 0, 
                 det_size = (640, 640), 
                 k: int = 5, 
                 similarity_threshold: float = 0.35):
        self.model = FaceAnalysis(name = model)
        self.model.prepare(ctx_id = ctx_id, det_size = det_size)
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.save_model_path = save_model_path
        self.knn_clf = None
        self.last_error = None

    def embed_face(self, img_array):
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        faces = self.__model.get(img)

        nums_of_people = len(faces)
        if len(faces) == 0:
            return [], nums_of_people
        return faces, nums_of_people

    def save_personal_data(self, personal_data, img_path):
        pass 

    def train(self, train_dir, imageManager, model_save_path = None, knn_algo = "ball_tree", verbose = True) -> Optional[None]:
        X = []
        y = []

        for class_dir in os.listdir(train_dir):
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            for img in imageManager.image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = imageManager.load_img_file(img)

                faces, nums_of_people = self.embed_face(image)
                if nums_of_people != 1:
                    if verbose:
                        self.last_error = f"Not suitbale image for training: {img}"
                else:
                    X.append(faces[0])
                    y.append(class_dir)
        
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors = self.k, algorithm = knn_algo, weights = "distance")
        knn_clf = knn_clf(X, y)
    
        if model_save_path is None:
            self.model_save_path = "./core/KNNClassifier.pkl"
            print(f"Chose model save path automatically: {self.model_save_path}")
        with open(model_save_path, "wb") as file:
            pickle.dump(knn_clf, file)
        self.knn_clf = knn_clf

    def predict(self, img_array):
        if self.save_model_path is None:
            self.model_save_path = "./core/KNNClassifier.pkl"
            print(f"Chose model save path automatically: {self.model_save_path}")
            
        if self.knn_clf is None:
            with open(self.save_model_path, "rb") as file:
                self.knn_clf = pickle.load(file)
        
        faces, nums_of_people = self.embed_face(img_array)
        if nums_of_people == 0:
            return []

        for face in faces:
            closest_distances = self.knn_clf.kneigbours(face, n_neighbors = self.k)
            are_matches = [closest_distances[0][i][0] <= self.similarity_threshold for i in range(len(faces))]
        return None