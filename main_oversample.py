#!/usr/bin/python
# -*- coding: utf-8

## IF702 Redes Neurais
import time
import math
import random
import numpy as np
import pandas as pd

from collections  import Counter
from treinamento  import Treinamento
from formatacao   import Formatacao
from oversampling import Oversampling

form = Formatacao()
form.read_database()
X_train, y_train = form.getTrain()
X_valid, y_valid = form.getValidation()
X_test, y_test   = form.getTest()

## Aplicando as funcoes de oversampling
sm = Oversampling()
methods, X_train_res, y_train_res = sm.ressample(X_train,y_train)
methods, X_valid_res, y_valid_res = sm.ressample(X_valid,y_valid)

## Treinamento da Rede
t = Treinamento()
t.define_rede() 
params = t.getRede()

text = []
text.append('CONFIGURACOES DA REDE:\nCamadada intermediaria: Nº Neuronios = '+str(params[0])+' | Funcao de ativacao = '+params[1])
text.append('\nCamadada de saída: Nº Neuronios = '+str(params[2])+' | Funcao de ativacao = '+params[3])
text.append('\nBatch: '+str(params[4])+' | Optimizer: '+params[5]+' | Funcao de perda = ' + params[6] + ' | Epocas: '+str(params[7])+' | Verboses: '+str(params[8])+'\n\n')

for i in range(len(methods)):
    text.append(methods[i])
    t.define_database(X_train_res[i], y_train_res[i], X_valid_res[i], y_valid_res[i], X_test, y_test)
    text.append(t.treinar())
    text.append('\n\n')

## Escrevendo Relatório
nomeRelatorio = 'relatorio_oversample_'+time.strftime("%d_%m_%y_")+time.strftime("%Hh_%Mm_%Ss")+'.txt'
ref_relatorio = open(nomeRelatorio, 'w')
for i in range(len(text)):
    ref_relatorio.writelines(text[i])
ref_relatorio.close()

print('Original dataset shape - train {}'.format(Counter(y_train)))
print('Original dataset shape - valid {}'.format(Counter(y_valid)))
print('Original dataset shape - test {}'.format(Counter(y_test)))
for i in range(len(y_train_res)):
    print('Resampled dataset shape - train {}'.format(Counter(y_train_res[i])))
for i in range(len(y_valid_res)):
    print('Resampled dataset shape - valid {}'.format(Counter(y_valid_res[i])))

for i in range(len(text)):
    print(text[i])
