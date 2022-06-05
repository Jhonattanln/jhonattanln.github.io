---
layout: post
title: "Machine Learning - Random Forest"
subtitle: "Analise classificatória para  risco de crédito- Python"
background: '/img/posts/Decision_Tree/Decision_Tree.jpg'
---
# Machine Learning - Random Forest

Análise de crédito é uma área extremamente importante para provedores de crédito, neste sentido o uso de **Inteligência Artificial** para ánalise de risco de quem toma o crédito é muito comum já que por ela é possível estimar bons e más pagadores.

A análise abaixo é feita através de um Dataset que replica informações de clientes de um banco disponibilizadas pelo Kaggle.

 ```python
### Importando bibliotecas
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile ## biblioteca para descompactar arquivo .zip
```
Os dados foram coletado da plataforma Kaggle e são disponibilizados via arquivo zipado, portanto devemos extrar os arquivos
```python
path = 'archive.zip' ## nome do arquivo .zip

## Usando ZipFile para extrair
with ZipFile(path, mode="r") as f:
    ## Listando os nomes dos arquivos
    file_names = f.namelist() 
    print(file_names)
    ## Extraindo arquivos
    f.extractall()
```
### Análise e manipulação de dados

```python
### Importando dados
customer = pd.read_csv('customer_data.csv')
customer.head()
```
!['Tabela'](/img/posts/Random_Forest/Tabela.png)
```python
### Checando estrutura dos dados 
customer.info()

### Checando dados da amostra
customer.info()
```
Com os código a cima é possível notar a 