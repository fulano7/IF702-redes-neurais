import math
import random
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

class Undersampling:
    def ressample(self, X, y):
        self.method = ['random under sampling']
        self.sX = []
        self.sY = []
            
        # Aplicando Random Over Sample
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_sample(X,y)
        
        self.sX.append(X_res)
        self.sY.append(y_res)
        
        return self.method, self.sX, self.sY
