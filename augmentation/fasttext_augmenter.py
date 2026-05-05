import fasttext
import random
from functools import lru_cache

class FastTextAugmenter:
    def __init__(self, model_path, sim_threshold=0.75, top_k=10):
        self.model = fasttext.load_model(model_path)
        self.sim_threshold = sim_threshold
        self.top_k = top_k

    @lru_cache(maxsize=10000)
    def get_candidates(self, word):
        try:
            neighbors = self.model.get_nearest_neighbors(word, k=self.top_k)
            return [w for sim, w in neighbors if sim >= self.sim_threshold]
        except:
            return []

    def augment_token(self, token, p=0.3):
        if random.random() > p:
            return token
        
        candidates = self.get_candidates(token)
        
        if not candidates:
            return token
        
        return random.choice(candidates)

    def augment_text(self, text, p=0.3):
        tokens = text.split()
        return " ".join([self.augment_token(t, p) for t in tokens])