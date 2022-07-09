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
### Base de dados:
A base de dados utilizada foi coletada da plataforma Economatica e contém:
- Fechamento do ativo: BOVA11
- Quantidade de negócios;
- Volume;
- Fechamento;
- Abertura;
- Mínima;
- Máxima;
- Médio.

```python
### Importando a base de dados
df = pd.read_excel('economatica.xlsx', skiprows=3, index_col=0, parse_dates=True)
df.rename(columns={'Volume$': 'Volume'}, inplace=True)
```
### Features:
A biblioteca **ta** foi utilizada para extrair as features da base de dados.

```python
df['Retornos'] = df.Fechamento.pct_change() ## retornos
df['Kama'] = ta.momentum.KAMAIndicator(close=df.Fechamento, window=21).kama() ## indicador Kama
df['ROC'] = ta.momentum.ROCIndicator(close=df.Fechamento, window=12).roc()
df['RSI'] = ta.momentum.RSIIndicator(close=df.Fechamento, window=14).rsi()
df['Stoch'] = ta.momentum.StochasticOscillator(close=df.Fechamento, high=df.Máximo, low=df.Mínimo, 
                                                window=14, smooth_window=3).stoch()
df['Chaikin_money'] = ta.volume.ChaikinMoneyFlowIndicator(high=df.Máximo, low=df.Mínimo, close=df.Fechamento, 
                                                          volume=df.Volume, window=20).chaikin_money_flow()
df['Force_index'] = ta.volume.ForceIndexIndicator(close=df.Fechamento, 
                                                  volume=df.Volume, window=13).force_index() 
df['Normal'] = (df.Fechamento - df.Mínimo) / (df.Máximo - df.Mínimo) 
```
### Tratando dados:
Houve a necessidade da limpeza de dados faltantes quando calculados os indicadores, como também a criação de targets para o modelo de classificação supervisionada