import math
import random
import numpy as np
import pandas as pd

class Formatacao:
    """ Leitura da base de dados e formataçao dos conjuntos """
    def union (self, A, B):
        v = []
        r = []
        for i in range(len(A)):
            aux = []
            for j in range(len(A[i])-1):
                aux.append(A[i][j])            
            v.append(aux)
            r.append(int(A[i][j+1]))

        for i in range(len(B)):
            aux = []
            for j in range(len(B[i])-1):
                aux.append(B[i][j])
            v.append(aux)
            r.append(int(B[i][j+1]))
            
        return v, r

    def read_database(self):
        """ Leitura e Limpeza dos Dados """
        data_set_c1 = pd.read_csv('data/redes_neurais_classe_1.csv', sep=';')
        data_set_c2 = pd.read_csv('data/redes_neurais_classe_2.csv', sep=';')

        ## data_set.drop_duplicates(inplace=True)  
        ## Remove exemplos repetidos
        ## Separando o data set em atributos dependentes (X = features) e independentes (y = classe). 
        xy1 = data_set_c1.iloc[:, :].values
        xy2 = data_set_c2.iloc[:, :].values

        ## Aqui dividimos o data set em treino, validação e teste.
        ## Treino: 50%, Validação: 25%, Teste: 25%
        random.shuffle(xy1)
        random.shuffle(xy2)
        c1_train = xy1[0:20000].tolist()
        c1_validation = xy1[20000:30000].tolist()
        c1_test = xy1[30000:40000].tolist()

        c2_train = xy2[0:350].tolist()
        c2_validation = xy2[350:525].tolist()
        c2_test = xy2[525:700].tolist()

        self.X_train, self.y_train = self.union(c1_train, c2_train)
        self.X_valid, self.y_valid = self.union(c1_validation, c2_validation)
        self.X_test, self.y_test   = self.union(c1_test, c2_test)

    def getTrain(self):
        """ retorna os conjuntos de treinamento """
        return np.asarray(self.X_train),np.asarray(self.y_train)

    def getValidation(self):
        """ retorna os conjuntos de treinamento """
        return np.asarray(self.X_valid),np.asarray(self.y_valid)

    def getTest(self):
        """ retorna os conjuntos de treinamento """
        return np.asarray(self.X_test),np.asarray(self.y_test)
