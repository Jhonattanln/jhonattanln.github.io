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
Com os código a cima é possível notar valores nulos no dataset, o que necessariamente deve ser tratado.
```python
print(customer.isnull().sum()) ### Contagem de valores nulos

#### Substituindo valores nulos
for col in customer.columns:
    if customer[col].isnull().sum() > 0:
        customer[col].fillna(customer[col].mean(), inplace=True)
```
### Processamento de dados

Em geral esses tipos de dados são desbalanceados, ou seja, terá bem mais observações de um respectivo dado dentro da amostra. Um exemplo de dados desbalanceados seria uma caixa de email, onde 99% dos emails que chagam são emails "limpos" (não nenhum tipo de má intensão) e apenas 1% dos emails são spam. Dessa maneira seria possível treinar um modelo de ML com esses dados que possuiria uma acurácia de 99%, neste caso ele poderia te recomendar todos os emails que chegam em sua caixa, juntamente com os emails maliciosos. Este é o problema de trabalhar com dados desbalanceados.
```python
### Contando targets
sns.countplot(x='label', data=customer, palette='hls')
plt.show()
```
!['Contagem'](/img/posts/Random_Forest/Contagem.png)

```python
### Criando variaveis dos dados
X = customer.drop(['label'], axis=1)
y = customer['label']
```
Uma alternativa para se trabalhar com dados desbalanceados seria seria excluir os targets de maior frequência e deixa-los na mesma quantidade, porém isso prejudica o modelo já que irá diminuir a base de dados e com isso o desempenho irá diminuir i.e. acurácia. Para evitar esse problema é muito comum o uso do SMOTE (_Synthetic Minority Oversampling Technique_) que basicamente é um algoritmo replica a amostra de menor frequência sinteticamente, ou seja, não é simplismente um copia e cola ela terá distinção nas features.
```python
### Aplicando SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
```
Podemos fazer a contagem novamente
```python
### Conatando dados
sns.countplot(y_res, palette='hls')
plt.show()
```
!['Contagem_2'](/img/posts/Random_Forest/Contage_1=2.png)

### Preparando Pipeline

Finalizada a parte de tratamento de dados partimos para criação do pipeline
```python
### Separando dados de treinameto
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
```
