---
layout: post
title: "Machine Learning - Bayesian Optimization"
subtitle: "Ajuste de hiperparâmetros- Python"
background: '/img/posts/Decision_Tree/Decision_Tree.jpg'
---
# Machine Learning - Bayesian Optimization

O ajuste de hiperpârametros em um modelo de **Machine Learning** é extremamente importante, já que pode elevar substancialmente o desempenho do modelo em aderência aos dados.
É muito comum ver em tutoriais de Machine Learning o uso do **Grid Search** para ajustar os hiperparâmetros, entranto há um problema de custo de processamento alto já que o Grid Search irá executar vários modelos e testar todo o intervalo de hiperparâmetros definido pelo usuário.
Em contra partida o **Bayesian Optimization** é um método de ajuste de hiperparâmetros que utiliza a tecnologia de **Bayesian Inference** para encontrar o melhor valor de hiperparâmetros para o modelo.

#### Estudo
Usaremos um estudo já abordado em um tutorial aqui no blog: [Trading Machine - Decision Tree](https://jhonattanln.github.io/2022/05/12/Decision_Tree.html)
Onde basicamente o objetivo é classificar se o retorno do ativo será positivou ou negativo ao longo dos dias.

### Bibliotecas utilizadas:
```python
import pandas as pd
import numpy as np
import ta 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt import space
from functools import partial
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
```