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

```python
df = df.dropna() ## excluindo valores nulos
X = df[[
        'Q Negs', 'Q Títs', 'Volume', 'Fechamento', 'Abertura', 'Mínimo', 
        'Máximo', 'Médio', 'Kama', 'ROC', 'RSI', 'Stoch', 'Chaikin_money', 
        'Force_index', 'Normal']] ## criando as features
y = np.where(df['Fechamento'].shift(-1) > df['Fechamento'], 1, -1) ## criando target
```
Através train_test_split foi criado a base de treinamento e teste
Houve a normalização dos dados para que os valores fossem entre 0 e 1 através do método StandardScaler

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
A ideia por trás do **Bayesian Optimization** é criar um modelo probabilistico usando um processo Gaussiano, ou seja, os valores assumidos da função seguem uma distribuição Gaussiana multivariada. Onde a covariância dos valores da função é dada por um Kernel GP entre os parâmetros.

>> [Bayesian Optimization](https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html)

```python
### Função de minimização para a Random Forest
def optimize(params, param_names, x, y):
    params = dict(zip(param_names, params))
    model = RandomForestClassifier(**params)
    kf = KFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = accuracy_score(ytest, preds)
        accuracies.append(fold_acc)

    return -1.0 * np.mean(accuracies)
```
```python
### Escolhendo os parâmetros para a Random Forest
parameters = [
    space.Integer(3, 15, name="max_depth"),
    space.Categorical(["gini", "entropy"], name="criterion"),
    space.Integer(100, 1000, prior="uniform", name="n_estimators"),
    space.Integer(2, 10, name="min_samples_split")
    ]

param_names = ['max_depth', 'criterion', 'n_estimators', 'min_samples_split']
```
```python
### Função de otimização para Bayes Optimization
optimization_function = partial(
    optimize,
    param_names=param_names,
    x=X_train,
    y=y_train
)
### Rodando o Bayes Optimization
result = gp_minimize(
    optimization_function, 
    dimensions=parameters,
    n_calls=15, 
    n_random_starts=10, 
    verbose=10
)
```
### Conferindo hiperparâmetros do modelo:
```python
print(dict(zip(param_names, result.x))) ## imprime os parametros
```
Agora que já descobrimos os hiperparâmetros que melhor se ajustam aos dados, é necessário rodar novamente o modelo com os hiperparâmetros desejados.

```python
rf = RandomForestClassifier(**dict(zip(param_names, result.x))) ## criando o modelo
rf.fit(X_train, y_train) ### treinando o modelo

### Testando o modelo
report = classification_report(y_test, rf.predict(X_test))
print(report)
```
Com o modelo já treinado é possível analisar o desempenho da estratégia
