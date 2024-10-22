import os
import re
import pickle 
import logging
import numpy as np
from PIL import Image 
from collections import Counter 
from typing import Dict, List, Optional, Tuple, Onion


class KNNClassifier:
    def __init__(self, k: int = 5, similarity_threshold: float = 0.35):
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.knn_clf = None

