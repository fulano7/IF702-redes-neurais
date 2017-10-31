import math
import random
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

class Treinamento:        
    def define_database(self, train_v, train_r, valid_v, valid_r, test_v, test_r):
        scaler = StandardScaler()
        self.dimension = len(train_v[0])
        self.X_train = scaler.fit_transform(train_v)
        self.y_train = train_r
        self.X_val = scaler.transform(valid_v)
        self.y_val = valid_r
        self.X_test = scaler.transform(test_v)
        self.y_test = test_r

    def getRede(self):
        rede = []
        rede.append(self.neuronios_inter)
        rede.append(self.activation_inter)
        rede.append(self.neuronios_saida)
        rede.append(self.activation_saida)
        rede.append(self.batch)
        rede.append(self.optimizer_func)
        rede.append(self.loss_func)
        rede.append(self.epocas)
        rede.append(self.verboses)
        return rede
       
        
    def define_rede(self, neuronios_inter = 16, activation_inter='tanh', neuronios_saida = 1, activation_saida = 'sigmoid',
                    batch = 64, optimizer_func = 'adam', loss_func='mean_squared_error', epocas = 100000, verboses = 0):
        self.neuronios_inter = neuronios_inter
        self.activation_inter = activation_inter
        self.neuronios_saida = neuronios_saida
        self.activation_saida = activation_saida
        self.batch = batch
        self.optimizer_func = optimizer_func
        self.loss_func = loss_func
        self.epocas = epocas
        self.verboses = verboses
        
        
    def extract_final_losses(self, history):
        """ Função para extrair o loss final de treino e validação.
            Argumento(s): history -- Objeto retornado pela função fit do keras.
            Retorno: Dicionário contendo o loss final de treino e de validação. """
        return {'train_loss': history.history['loss'][-1], 'val_loss': history.history['val_loss'][-1]}

    def treinar(self):
        try:        
            # Cria o esboço da rede.
            classifier = Sequential()

            # Adiciona a primeira camada escondida contendo 16 neurônios e função de ativação tangente 
            # hiperbólica. Por ser a primeira camada adicionada à rede, precisamos especificar a 
            # dimensão de entrada (número de features do data set).
            classifier.add(Dense(self.neuronios_inter, activation = self.activation_inter, input_dim = self.dimension))

            # Adiciona a camada de saída. Como nosso problema é binário, só precisamos de 1 neurônio 
            # e função de ativação sigmoidal. A partir da segunda camada adicionada, keras já consegue 
            # inferir o número de neurônios de entrada (nesse caso 16) e nós não precisamos mais 
            # especificar.
            classifier.add(Dense(self.neuronios_saida, activation = self.activation_saida))

            # Compila o modelo especificando o otimizador, a função de custo, e opcionalmente métricas 
            # para serem observadas durante o treinamento.
            classifier.compile(optimizer = self.optimizer_func, loss = self.loss_func)

            # Treina a rede, especificando o tamanho do batch, o número máximo de épocas, se deseja 
            # parar prematuramente caso o erro de validação não decresça, e o conjunto de validação.
            history = classifier.fit(self.X_train, self.y_train, batch_size = self.batch, epochs = self.epocas, 
                                     callbacks=[EarlyStopping()], validation_data=(self.X_val, self.y_val))
                    
            ## Usar a nossa rede para fazer predições e computar métricas de desempenho
            ## Mais métricas de desempenho: http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics"

            ## Fazer predições no conjunto de teste
            y_pred = classifier.predict(self.X_test)
            y_pred_class = classifier.predict_classes(self.X_test, verbose = self.verboses)

            text = []
            ## Matriz de confusão
            cmatrix = confusion_matrix(self.y_test, y_pred_class)
            text.append('\n\tMatriz de confusão:')
            strg = "\n\t"
            for i in range(len(cmatrix)):
                strg += "| "
                for j in range(len(cmatrix[i])):
                    strg += str(float(cmatrix[i][j]))+ ' '
                strg += '|\n\t'
            text.append(strg)

            ## Computar métricas de desempenho
            losses = self.extract_final_losses(history)
            
            text.append("{metric:<18}{value:.4f}".format(metric="\n\tTrain Loss:", value=losses['train_loss']))
            text.append("{metric:<18}{value:.4f}".format(metric="\n\tValidation Loss:", value=losses['val_loss']))
            text.append("{metric:<18}{value:.4f}".format(metric="\n\tAccuracy:", value=accuracy_score(self.y_test, y_pred_class)))
            text.append("{metric:<18}{value:.4f}".format(metric="\n\tRecall:", value=recall_score(self.y_test, y_pred_class)))
            text.append("{metric:<18}{value:.4f}".format(metric="\n\tPrecision:", value=precision_score(self.y_test, y_pred_class)))
            text.append("{metric:<18}{value:.4f}".format(metric="\n\tF1:", value=f1_score(self.y_test, y_pred_class)))
            text.append("{metric:<18}{value:.4f}".format(metric="\n\tAUROC:", value=roc_auc_score(self.y_test, y_pred)))

##            print('Matriz de confusão:\n')
##            print(cmatrix)
##            print("{metric:<18}{value:.4f}".format(metric="Train Loss:", value=losses['train_loss']))
##            print("{metric:<18}{value:.4f}".format(metric="Validation Loss:", value=losses['val_loss']))
##            print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy_score(self.y_test, y_pred_class)))
##            print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall_score(self.y_test, y_pred_class)))
##            print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision_score(self.y_test, y_pred_class)))
##            print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1_score(self.y_test, y_pred_class)))
##            print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=roc_auc_score(self.y_test, y_pred)))
            
            return text

        except:
            print ('Ocorreu algum error...')
            return []
        
