import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        X=np.array(x)
        gamma=np.array(gamma)
        rms = np.sqrt(np.mean(X ** 2) + eps)
        x_hat = x / rms

        return np.round(gamma * x_hat, 4).tolist()

