---
layout: post
title: "Econometria - Teorema do limite central"
subtitle: "Aplicação do teorema do limite central - R"
background: '/img/posts/Teorema_limite\normal.jpg'
---
## Teorema do Limite Central

O Teorema do Limite Central ou TLC é um dos conceitos mais importantes da estatística moderna e utilizado para diversos problemas com amostragem.

**Teorema do Limite Central:** _Dada uma variável X, i.i.d (independente e identicamente distribuida), com média µ e variância σ², a média amostral de X segue uma distribuição *normal* desde que a amostra seja suficientemente grande_
> Definão tirada do livro _Estatística e introdução a econometria - Sartoris_

Quanto maior o número de observações na amostra da média, maior será a convergência da distribuição para a normal 

### Simulação

```r
### Importando bibliotecas
library(tibble)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggthemes)
```
Para a simulação foram utilizadas diferentes distribuições

Distribuição |
-------------|
Exponencial  |
Uniforme     |
t-Student    |
f-Fisher     |

```r
### Criando dodos para análise
n <- 5000 # número da amostra
m <- 5000 # número da amostra que serão tiradas as medias

sim <- tibble(indice = 1:n, ### tibble para entrada dos dados
              exponencial = double(length = n),
              uniforme = double(length = n),
              tStudent = double(length = n),
              fFisher = double(length = n))
### Estruturando as médias

set.seed(1234)
for(i in 1:n) {
  sim$exponencial[i] <- rexp(n = m) %>% mean()
  sim$uniforme[i] <- runif(n = m) %>% mean()
  sim$tStudent[i] <- rt(n = m, df = 2) %>% mean()
  sim$fFisher[i] <- rf(n = m, df1 = 2, df2 = 4) %>% mean()

}

```

O próximo passo será normalizar a média de X, assim temos:
![Teorema](/img/posts/Teorema_limite/Captura_1.png)
<img src="/img/posts/Teorema_limite/Captura_1.png" alt="drawing" width="50"/>

Portanto, o TLC terá média 0 e variância 1
![Teorema](/img/posts/Teorema_limite/Captura_2.png)