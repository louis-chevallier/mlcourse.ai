
from utillc import *
import numpy as np
from matplotlib import pyplot as plt

class MeanClassifier :
    """
    ne gère que le cas binaire ..
    """
    def __init__(self) :
        pass

    def fit(self, X, y) :
        EKOX(X.shape)        
        EKOX(y.shape)
        EKOX(np.mean(y))
        #plt.hist(y); plt.show()
        self.pos = X[y> 0.5]
        self.neg = X[y<= 0.5]
        EKOX(self.pos.shape)
        EKOX(np.mean(self.pos, axis=0).shape)
        self.posm = np.mean(self.pos, axis=0)
        self.negm = np.mean(self.neg, axis=0)
        pass

    def predict(self, X) :
        EKOX(X.shape)
        N, D = X.shape

        EKOX((X-self.posm).shape)
        EKOX(np.linalg.norm(X-self.posm, axis=1).shape)
        dpos = np.linalg.norm(X-self.posm, axis=1)
        dneg = np.linalg.norm(X-self.negm, axis=1)

        EKOX(dpos.shape)
        EKOX((dpos-dneg).shape)
        EKOX((dpos > dneg).shape)
        
        rep = np.zeros(N)
        rep = dneg > dpos
        return rep
