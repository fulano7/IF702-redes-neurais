import math
import random
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

class Oversampling:
    def ressample(self, X, y):
        self.method = ['random over sampling','regular SMOTE over sampling', 'borderline1 SMOTE over sampling', 'borderline2 SMOTE over sampling']
        self.sX = []
        self.sY = []
            
        # Alicando Random Over Sample
        ros = RandomOverSampler()
        X_res, y_res = ros.fit_sample(X,y)
        self.sX.append(X_res)
        self.sY.append(y_res) 

	# - Funcao de sampling
        # Aplicando SMOTE regular e borderline
        kinds = ['regular']#, 'borderline1', 'borderline2']
        sm = [SMOTE(kind = k) for k in kinds]
        for i in range(len(sm)):
            X_res, y_res = sm[i].fit_sample(np.asarray(X_res), np.asarray(y_res))
            self.sX.append(X_res)
            self.sY.append(y_res)

        return self.method, self.sX, self.sY
