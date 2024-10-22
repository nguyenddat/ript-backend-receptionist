import os
import cv2
import base64
import pickle
import numpy as np
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
                 k: int = 3, 
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
        faces = self.model.get(img)

        nums_of_people = len(faces)
        if nums_of_people == 0:
            return [], nums_of_people
        return [face.embedding for face in faces], nums_of_people

    def save_personal_data(self, personal_data, img_path):
        pass 

    def train(self, train_dir, imageManager, model_save_path = None, knn_algo = "ball_tree", verbose = True) -> Optional[None]:
        self.model_save_path = model_save_path
        X = []
        y = []

        for class_dir in os.listdir(train_dir):
            temp = os.path.join(train_dir, class_dir)
            if not os.path.isdir(temp):
                continue

            for img in imageManager.image_files_in_folder(temp):
                image = imageManager.load_img_file(img)

                faces, nums_of_people = self.embed_face(image)
                if nums_of_people != 1:
                    if verbose:
                        self.last_error = f"Not suitbale image for training: {img}"
                else:
                    X.append(faces[0])
                    y.append(class_dir)
        
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors = self.k, algorithm = knn_algo, weights = "distance")
        knn_clf = knn_clf.fit(X, y)

        print(True)
        if self.model_save_path is None:
            self.model_save_path = "./core/KNNClassifier.pkl"
            print(f"Chose model save path automatically: {self.model_save_path}")
        with open(self.model_save_path, "wb") as file:
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
    
        results = []
        for face in faces:
            face_embedding = face.reshape(1, -1)

            distances, indices = self.knn_clf.kneigbors(face_embedding, n_neighbors = self.k)
            distances = distances[0]

            closest_labels = [self.knn_clf.classes_[idx] for idx in indices[0]]

            weights = 1 / (distances + 1e-6)
            weights = weights / np.sum(weights)

            label_weights = {}
            for label, weight in zip(closest_labels, weights):
                label_weights[label] = label_weights.get(label, 0) + weight
            
            predict_label = max(label_weights.items(), key = lambda x: x[1])[0]
            if np.mean(distances) > self.similarity_threshold:
                predict_label = "unknown"
            results.append(predict_label)
        return results