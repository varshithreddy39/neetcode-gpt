import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        s=z-np.max(z)
        exp=np.exp(s)
        return np.round(exp/np.sum(exp),4)
        
