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

Portanto, o TLC terá média 0 e variância 1
![Teorema](/img/posts/Teorema_limite/Captura_2.png)

Podemos então plotar um gráfico da distribuição de qualquer um das diferentes distribuições

```r
### Plotando distribuição 

janela %>%
  ggplot(aes(x = fFisher))+ # distribuiÃ§Ã£o Fisher
  geom_histogram(aes(y = ..density..),bins = 70, fill = 'black', alpha = 0.8)+
  geom_density(size = 1.5, alpha = 0.9, color = 'red')+
  theme_hc()+
  scale_colour_hc()

ecdf_fisher <- ecdf(janela$fFisher)

plot(ecdf_fisher)
```
![Distribuição](/img/posts/Teorema_limite/Rplot_distribuicao.png)
![ECDF](/img/posts/Teorema_limite/Rplot_ecdf.png)

Agora vamos fazer o teste de _Kolmogorov-Smirnov_ para analisar se os dados seguem uma distribuição normal e também notar a dinâmica dos valores quando aumentamos a amostra .

```r
for(i in 3:nrow(testes)) {
  
  janela <- sim %>% 
    filter(indice <= i) %>% 
    transmute(uniforme = (uniforme - mean(uniforme)/sd(uniforme)),
              tStudent = (tStudent - mean(tStudent))/sd(tStudent),
              fFisher = (fFisher - mean(fFisher))/sd(fFisher))

  ### Rodando teste de Kolmogorov-Smirnov
  
  testes$Puni[i] <- ks.test(x = janela$uniforme, 'pnorm')$p.value
  testes$PtStu[i] <- ks.test(x = janela$tStudent, 'pnorm')$p.value
  testes$PfFis[i] <- ks.test(x = janela$fFisher, 'pnorm')$p.value
  
}
### Gráfico do p_value

testes %>%
  pivot_longer(Puni:PfFis,
               names_to = "distro",
               values_to = "p") %>%
  ggplot(aes(x = indice, color = distro, y = p))+
  geom_line(size=1.2, alpha = 0.8)+
  theme_hc()+
  scale_colour_hc()+
  scale_y_continuous(label = scales::percent)
```
![P_value](/img/posts/Teorema_limite/Rplot_pvalue.png)