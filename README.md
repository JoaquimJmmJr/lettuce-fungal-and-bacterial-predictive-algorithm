# Algoritmo preditivo e modelo de classificação (CNN) para combate ao Míldio (Bremia lactucae) do alface

Sistema de antecipação de ocorrência de doenças com base em variáveis ambientais utilizando um sistema determinístico que determina o grau de risco de surgimento do Míldio e, quando ele sobe, o robô fotografa para o modelo de rede neural fazer a validação visual dos primeiros sintomas, permitindo intervenção precoce, redução do uso de insumos e recuperação do cultivo.

**Objetivo**: Ajudar no combate preventivo ao Míldio (_Bremia lactucae_), uma das doenças mais destrutivas e uma das maiores ameaças na cultura da alface, favorecida por:
- Alta umidade relativa (> 90%)
- Temperaturas amenas (10–20 °C)
- Período prolongado de exposição (≥ 6 horas)

**Escopo**: O projeto limita-se inicialmente ao Míldio (_Bremia lactucae_) por ser a doença mais limitante à cultura da alface (COSTA; ZAMBOLIM; VENTURA, 2007), por apresentar a maior disponibilidade de dados — datasets públicos, artigos acadêmicos que comprovam a eficácia do tratamento com Luz UV-C e fontes bibliográficas que descrevem as condições ambientais para seu surgimento — e pelo fato de que a dosimetria UV-C é específica para cada patógeno, não sendo diretamente transferível para outras doenças sem validação experimental própria. Estudos demonstram que a aplicação noturna de Luz UV-C é crítica para a eficácia do tratamento, pois evita a fotorreativação dos esporângios do fungo (SUTHAPARAN et al., 2012), o que torna o horário de aplicação uma variável operacional a ser integrada ao modelo junto às condições ambientais quando o sistema estiver em ambiente real.

---

## Sumário
1. [Algoritmo preditivo do Míldio usando um sistema determinístico](#1-algoritmo-preditivo-do-míldio-usando-um-sistema-determinístico)
2. [Modelo de classificação do Míldio usando CNN (Deep Learning)](#2-modelo-de-classificação-do-míldio-usando-cnn-deep-learning)
3. [Arquitetura do sistema](#3-arquitetura-do-sistema)
4. [Fluxo de funcionamento](#4-fluxo-de-funcionamento)
5. [Como executar o projeto](#5-como-executar-o-projeto)
6. [Melhorias futuras](#6-melhorias-futuras)
7. [Referências](#7-referências)

---

## 1. Algoritmo preditivo do Míldio usando um sistema determinístico

### 1.1 Justificativa da escolha do algoritmo

Com base no que a literatura agronômica descreve para o Míldio (_Bremia lactucae_), o sistema consegue calcular o risco com base em condições ambientais contínuas ao longo do tempo e identificar condições favoráveis à doença antes do surgimento visível dos sintomas. Um sistema determinístico é aquele em que, dada uma entrada, a saída é sempre a mesma e totalmente previsível — não há aprendizado, probabilidade ou parâmetros ajustáveis. As regras foram extraídas diretamente da literatura agronômica (limiares de temperatura, umidade e duração), o que torna o sistema confiável, explicável e defensável sem depender de dados históricos de ocorrência da doença. Para esse problema específico, isso é uma vantagem: um modelo de Machine Learning exigiria um histórico de registros rotulados ("deu míldio / não deu") que ainda não existe. O sistema pode evoluir futuramente para aprendizado de máquina (exemplo: regressão logística ou árvores de decisão) à medida que dados reais forem coletados pelo robô.

---

### 1.2 Regra principal (Míldio)

```
Se:
  umidade relativa > 90%
  E temperatura entre 10–20 °C
  Por um período contínuo ≥ 6 horas

→ RISCO ALTO
```

#### Classificação de risco

| Condição                           | Risco    |
| ---------------------------------- | -------- |
| ≥ 6h contínuas em condição crítica | 🔴 Alto  |
| 3h – 6h                            | 🟡 Médio |
| < 3h                               | 🟢 Baixo |

>Os limiares de 3 e 6 horas foram definidos com base nos estudos de Foroud et al. (2018), que demonstraram que apenas 2 horas de molhamento foliar podem ser suficientes para infecção em condições ótimas, e de Smith et al. (2021), que indicam 5 a 7 horas como referência para infecção em condições subtropicais. O limiar inferior de 3 horas oferece margem preventiva, enquanto o limiar superior de 6 horas sinaliza risco iminente de infecção estabelecida.

---

### 1.3 Janela temporal

As métricas centrais do modelo são o maior bloco contínuo de horas em condição crítica nas últimas 24 horas e o intervalo de varredura a cada 30 minutos. A escolha das 24 horas captura a essência epidemiológica do míldio de que não basta que UR e temperatura atinjam os limiares pontualmente, é a duração sustentada dessas condições que determina se a infecção terá tempo de se estabelecer [Foroud et al., 2018; Smith et al., 2021]. E a adoção do intervalo de 30 minutos representa o equilíbrio entre a resolução temporal, ou seja, o quão detalhadamente o sistema acompanha as mudanças ao longo do tempo, o consumo de energia, o processamento de dados e tempo de operação do robô para acompanhar e capturar as mudanças dos limiares epidemiológicos que favorecem o desenvolvimento da doença.

- A análise é baseada nas últimas 24 horas (`df[col_ts].max() - pd.Timedelta(hours=24)`)
- Cálculo executado a cada 30 minutos (`intervalo_min = 30`)

> O sensor SHT31 realizará leituras a cada varredura salvando a média, os máximos de UR e o tempo contínuo crítico (UR>90%) como processamento; ao final, o ESP32 publica esses dados via protocolo serial UART à Raspberry Pi que os recebe diretamente e os persiste no banco de dados MySQL. A duração é calculada pela diferença real entre timestamps consecutivos, garantindo robustez a leituras irregulares sem necessidade de interpolação.
---

### 1.4 Estrutura dos dados

Formato da tabela (CSV ou banco):
```
timestamp,temp,umidade,solo
2026-04-25 00:00:00,18,92,65
2026-04-25 00:30:00,17,94,66
```

#### Campos:
- `timestamp`: data e hora da leitura
- `temp`: temperatura (°C)
- `umidade`: umidade relativa (%)
- `solo`: umidade do solo (%)

---

### 1.5 Tecnologias utilizadas

| Camada                  | Tecnologia                                |
|-------------------------|-------------------------------------------|
| Backend / processamento | Python (Pandas + NumPy)                   |

#### Implementações futuras com dados reais:

| Hardware (IoT) | ESP32 + Sensores de temperatura e umidade |
|---|---|
| Banco de dados | MySQL     |
| Visualização   | Streamlit |

---

## 2. Modelo de classificação do Míldio usando CNN (Deep Learning)

### 2.1 Justificativa da escolha do modelo

Treinar uma rede neural convolucional (CNN) do zero exige **dezenas de milhares de imagens** para aprender a detectar formas, texturas e padrões visuais básicos. Datasets agrícolas raramente têm esse volume. O **Transfer Learning** resolve isso reaproveitando um modelo já treinado na **ImageNet** — base com 1,2 milhão de imagens e 1000 classes que já sabe detectar bordas, texturas, gradientes de cor e formas complexas (DENG et al., 2009). Com isso, só é preciso ensinar a parte final do modelo: dos padrões que ele já conhece, quais indicam míldio. Na prática, é possível chegar a bons resultados com **centenas de imagens ao invés de dezenas de milhares**, e o treinamento leva minutos em vez de horas. Esse modelo é acionado para fazer a validação visual quando o algoritmo determinístico detecta risco alto, confirmando se os sintomas já são visíveis na planta antes de acionar o tratamento com Luz UV-C.

---

### 2.2 Justificativa da escolha da arquitetura

Existem várias arquiteturas disponíveis como ResNet, VGG, InceptionV3, MobileNet etc. O EfficientNetB0 foi escolhido por três razões concretas para esse caso:

1. **Melhor acurácia por parâmetro**: o EfficientNet foi projetado com escala balanceada de profundidade, largura e resolução da rede (_compound scaling_). O resultado é que o B0 supera o ResNet50 em acurácia na ImageNet usando 5× menos parâmetros (TAN; LE, 2019).

| Modelo          | Parâmetros | Acurácia ImageNet |
|-----------------|------------|-------------------|
| VGG16           | 138M       | 71%               |
| ResNet50        | 25M        | 76%               |
| **EfficientNetB0** | **5.3M** | **77%**           |
| MobileNetV2     | 3.4M       | 72%               |

2. **Funciona bem com imagens de textura**: o Míldio é essencialmente uma doença de textura foliar — manchas amareladas, esporulação branca na face inferior da folha, alteração de cor (COSTA; VENTURA; LIMA, 2012). O EfficientNet captura bem padrões locais de textura, que é exatamente o sinal visual que diferencia uma folha doente de uma saudável.

3. **Leve o suficiente para o Colab gratuito**: o B0 treina confortavelmente na GPU T4 do Google Colab sem estourar memória, mesmo com batch size 32 e imagens 224×224. Versões maiores como B4 ou B7 dariam mais acurácia, porém exigiriam redução do batch ou uso do Colab Pro.

---

### 2.3 Estratégia de Transfer Learning

O EfficientNetB0 original foi treinado para classificar 1000 categorias de objetos do mundo real. Ao ser adaptado para este problema, duas partes do modelo foram tratadas de forma diferente:

- **Camadas iniciais — mantidas congeladas**: detectam elementos primitivos e universais como bordas, contrastes e gradientes de cor. Esse conhecimento é diretamente reutilizável para analisar folhas, pois detectar a textura do míldio exige as mesmas habilidades primitivas que detectar a textura de qualquer outro objeto.

- **Cabeça de classificação — substituída**: as últimas camadas originais do EfficientNetB0, responsáveis por classificar entre as 1000 categorias do ImageNet, foram removidas e substituídas por uma nova sequência de camadas densas treinadas do zero:
  ```
  GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.4)
                         → Dense(128, relu) → Dropout(0.3)
                         → Dense(1, sigmoid)   ← saída binária
  ```
  A camada de saída tem **1 neurônio com ativação sigmoid** porque o problema é binário: a função retorna um valor entre 0 e 1, interpretado como a probabilidade de a folha ser Healthy. Valores abaixo de 0.5 são classificados como Downey Mildew.

O treinamento ocorre em duas fases para evitar o *catastrophic forgetting* — a destruição do conhecimento adquirido no ImageNet antes que a nova cabeça aprenda algo útil:

| Fase | O que acontece | Learning Rate |
|------|---------------|---------------|
| 1 — Base congelada | Apenas a cabeça de classificação é treinada | `1e-3` |
| 2 — Fine-tuning | As últimas 30 camadas do backbone são descongeladas e ajustadas suavemente | `1e-5` |

---

### 2.4 Dataset e balanceamento

#### Composição do dataset de treinamento:
| Fonte (Roboflow Universe) | Classe | Imagens | 
| --- | --- | --- |
|2G Care Spice — Lettuce Folder X3 | Downy Mildew | 673 | 
|2G Care Spice — Lettuce Folder X3 | Healthy      | 569 |
|YSSP — Lettuce Detections         | Downy Mildew | 179 |
|LetCare — Lettuce Care            | Downy Mildew | 230 |
|Lettuce Disease — Lettuce Disease | Healthy | 1.087 |
|Total Downy Mildew | --- | 1.082 |
|Total Healthy | --- | 1.656 |
|**Total geral** | --- |**2.738** | 

#### Particionamento do dataset para treinamento:
| Partição | Imagens | Divisão (%) |
|----------|---------|-------------|
| Treino   | 1.918   | 70%         |
| Validação| 410     | 15%         |
| Teste    | 410     | 15%         |

**Balanceamento com `class_weight`**: como o dataset é desbalanceado (proporção de 1,53 imagens Healthy para cada 1 Downey Mildew), foi utilizada a função `compute_class_weight(class_weight='balanced')` do scikit-learn (PEDREGOSA et al., 2011). Ela atribui um peso de penalidade maior aos erros na classe minoritária: errar uma Downey Mildew custa 1,53× mais ao modelo do que errar uma Healthy, impedindo que ele "aposte sempre em Healthy" para maximizar acurácia.

---

### 2.5 Limitações e resultados observados

Os resultados do treinamento atual revelam limitações que precisam ser endereçadas antes da implantação em ambiente real:

#### Métricas obtidas (conjunto de teste)

| Métrica | Resultado | Interpretação |
|---------|-----------|---------------|
| AUC     | 0,588     | Próximo de 0,5 (classificação aleatória), indicando que o modelo mal separou as duas classes |
| Matriz de confusão | 100% das Downey Mildew classificadas como Healthy | O modelo aprendeu a classificar tudo como a classe majoritária |

> **AUC** (_Area Under the ROC Curve_) mede a capacidade do modelo de separar as classes em todos os thresholds possíveis. Um valor de 1,0 é perfeito; 0,5 é equivalente a um chute aleatório.

#### Causa identificada: overfitting por baixa representatividade do dataset

O dataset de treinamento apresenta características visuais que não correspondem ao cenário real de operação do robô:

- A maioria das imagens tem **fundo branco**, o que facilita a segmentação mas não existe no ambiente de cultivo indoor
- Algumas imagens têm **marcas d'água, pratos ou contextos artificiais**
- No cenário real, o robô fotografará folhas **sob iluminação artificial**, com as plantas **justapostas** umas às outras, sem isolamento visual

Como o modelo aprendeu a usar o fundo branco como pista para classificação, ao receber imagens no ambiente real — sem fundo branco — tende a falhar. O modelo memorizou características do dataset que não aparecem no mundo real e perdeu a capacidade de generalização: **overfitting** (GOODFELLOW; BENGIO; COURVILLE, 2016).

#### Melhoria de curto prazo: ajuste do threshold

O threshold padrão de 0,5 exige 50% de confiança para classificar uma imagem como Downey Mildew. Reduzir esse valor (ex.: para 0,3) faz o modelo classificar como Downey com apenas 30% de confiança, aumentando o número de detecções positivas e reduzindo os falsos negativos — o erro mais crítico em um sistema de detecção de doenças.

#### Melhoria estrutural necessária: novo dataset

A solução definitiva é substituir ou complementar o dataset com imagens coletadas no próprio ambiente de operação do robô (iluminação indoor, folhas juntas, sem fundo branco), alinhando a distribuição de treinamento com a distribuição real.

---

### 2.6 Retreinamento incremental

O modelo exportado no formato `.keras` preserva a arquitetura completa, os pesos aprendidos e o estado do optimizer, permitindo que o treinamento seja retomado a qualquer momento com novos dados sem reiniciar do zero. O notebook inclui uma célula dedicada de retreinamento incremental que:

1. Carrega o modelo salvo anteriormente
2. Aceita o dataset atualizado com as novas imagens nas mesmas subpastas de classe
3. Descongela apenas as últimas 30 camadas para atualizar as representações sem apagar o aprendizado anterior
4. Treina com `lr=1e-5` por até 10 épocas
5. Salva o modelo atualizado com versionamento (`_v2`, `_v3` etc.)

| Situação | Recomendação |
|---|---|
| Adicionou ~20% mais imagens | Retreine com a célula incremental (`epochs=10`, `lr=1e-5`) |
| Dobrou o dataset | Retreine do zero com `construir_modelo()` |
| Adicionou nova classe | Retreine do zero — a arquitetura de saída muda |

---

## 3. Arquitetura do sistema

```
ESP32 → transmite dados via UART →  MySQL (tabela temporal)
                                      ↓
                              Script Python (a cada 30 min)
                                      ↓
                              Calcula risco ambiental
                                      ↓
                            ┌── Risco ALTO? ───┐
                            Não               Sim
                            ↓                  ↓
                        Aguarda        Robô fotografa a planta
                                               ↓
                                      Modelo CNN classifica
                                      (Healthy / Downey Mildew)
                                               ↓
                                      Salva resultado / dashboard
                                               ↓
                                      Aciona tratamento com Luz UV-C
```

## 4. Fluxo de funcionamento

1. Sensores coletam dados ambientais a cada 30 minutos
2. Dados são enviados ao banco (MySQL)
3. Script Python:
   - Lê as últimas 24h
   - Identifica condições críticas de temperatura e umidade
   - Calcula blocos contínuos de exposição
4. Sistema classifica o risco ambiental (Baixo / Médio / Alto)
5. Quando o risco sobe para **Alto**, o robô é acionado para fotografar as plantas
6. O modelo CNN classifica as imagens capturadas
7. Se confirmada a presença de Míldio, o tratamento com **Luz UV-C noturna** é aplicado nas regiões afetadas

---

## 5. Como executar o projeto

### Clonar o repositório
```bash
git clone <repo>
cd <repo>
```

### Instalar dependências
```bash
pip install pandas numpy
```

### Utilizar um dataset de teste
Crie um arquivo `dados_sensores.csv` com o seguinte formato:
```
timestamp,temp,umidade,solo
2026-04-25 00:00:00,18,92,65
2026-04-25 00:30:00,17,94,66
```

#### Recomendado:
- Intervalo de 30 minutos
- 2 a 5 dias de dados
- Incluir períodos com alta umidade (> 90%)

### Executar o script
```bash
python modelo-deterministico-mildio.py
```

### Saída esperada
```
Horas contínuas em condição crítica: 6.5h
Nível de risco de Míldio: ALTO 🔴
```

---

## 6. Melhorias futuras

### Modelo determinístico → Machine Learning
Estratégia de evolução quando houver dados históricos reais de ocorrência:
1. **Regressão logística**: calcula a probabilidade de cada nível de risco sem ficar preso a limiares rígidos
2. **Random Forest**: funciona bem com poucos dados e aprende padrões não óbvios entre as variáveis (médio prazo)
3. **LSTM (séries temporais)**: aprende padrões temporais complexos ao longo do tempo (projeto avançado)

### Modelo CNN — melhorias prioritárias
- **Novo dataset com imagens do ambiente real**: coletar imagens das plantas diretamente no ambiente indoor de operação do robô, eliminando o problema de overfitting por fundo branco e condições artificiais
- **Augmentação de fundo**: aplicar técnicas de substituição de fundo (_background augmentation_) para simular diferentes condições de captura durante o treinamento
- **Ajuste de threshold**: reduzir o threshold de classificação de 0,5 para ~0,3, priorizando a redução de falsos negativos (plantas doentes classificadas como saudáveis)

### Integração UV-C e modelo unificado
Quando o sistema estiver em ambiente real, será necessário explorar a integração entre os limiares ambientais de surgimento da doença e os parâmetros de dosimetria UV-C em um modelo unificado, considerando o horário noturno de aplicação como variável operacional crítica.

### Expansão para outras doenças
A inclusão de outras doenças exigirá, para cada uma:
- Artigos acadêmicos que comprovem a eficácia do tratamento com Luz UV-C para aquele patógeno específico
- Datasets públicos disponíveis (Roboflow, Kaggle, PlantVillage etc.)
- Fontes bibliográficas com as condições ambientais de surgimento
- Validação da dosimetria UV-C, que é diferente para cada tipo de patógeno

Doenças com potencial de inclusão futura:
1. Mancha bacteriana (_Bacterial Blight_)
2. Ácaro-rajado (_Tetranychus urticae_)
3. Mosca-branca (_Trialeurodes vaporariorum / Bemisia tabaci_)
4. Mofo-cinzento (_Botrytis cinerea_)
5. Cercosporiose (_Cercospora longissima_)

### Outras funcionalidades
- Mapa de calor da plantação
- Histórico de risco
- Alertas em tempo real

---

## 7. Referências
2G CARE SPICE. **Lettuce Folder X3 Dataset: lettuce downy mildew and healthy leaves**. Roboflow Universe, 2024. Disponível em: https://universe.roboflow.com/2g-care-spice/lettuce-folder-x3. Acesso em: abr. 2026. 

AARROUF, J.; URBAN, L. **Flashes of UV-C light: an innovative method for stimulating plant defences**. PLoS ONE, v. 15, n. 7, e0235918, 2020.

COSTA, H.; VENTURA, J. A.; LIMA, I. M. Doenças da alface no estado do Espírito Santo: diagnose e manejo. In: COSTA, H. (Ed.). **Cultura da alface**: aspectos fitossanitários. Vitória: Incaper, 2012. cap. 6, p. 71–92.

COSTA, H.; ZAMBOLIM, L.; VENTURA, J. A. Doenças de hortaliças que se constituem em desafios para o controle. In: ZAMBOLIM, L. et al. (Eds.). **Manejo Integrado de Doenças e Pragas: hortaliças**. Visconde do Rio Branco: Suprema, 2007. p. 319–348.

DENG, J. et al. **ImageNet: a large-scale hierarchical image database**. In: IEEE CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION, 2009, Miami. Proceedings [...]. New York: IEEE, 2009. p. 248–255.

FOROUD, N. A. et al. **Infection conditions for downy mildew of lettuce caused by Bremia lactucae**. Canadian Journal of Plant Pathology, v. 40, n. 3, p. 376–386, 2018.

GÉRON, A. **Mãos à obra: aprendizado de máquina com Scikit-Learn, Keras e TensorFlow**. 3. ed. Rio de Janeiro: Alta Books, 2025.

GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. **Deep Learning**. Cambridge: MIT Press, 2016.

INSTITUTO BIOLÓGICO. **Guia de Sanidade Vegetal**. Versão 1.0. São Paulo: Instituto Biológico, [s.d.]. Disponível em: https://www.sica.bio.br/guiabiologico/busca_culturas_resultado_ok.php?Id=13&Vlt=3. Acesso em: 5 maio 2026.

KUSHALAPPA, A. C. **BREMCAST: development of a system to forecast the risk levels of downy mildew on lettuce based on sporulation**. International Journal of Pest Management, v. 47, n. 1, p. 1–5, 2001.

LETCARE. **Lettuce Care Dataset: downy mildew on lettuce**. Roboflow Universe, 2024. Disponível em: https://universe.roboflow.com/letcare/lettuce-care-aadyx. Acesso em: abr. 2026. 

LETTUCE DISEASE DATASET. **Lettuce Disease: healthy lettuce leaves**. Roboflow Universe, 2024. Disponível em: https://universe.roboflow.com/lettuce-gjt5u/lettuce-disease-uuxm6. Acesso em: abr. 2026.

LOPES, C. A. et al. Principais doenças e pragas da alface. In: ______. **Alface de A a Z**. Brasília: Embrapa Hortaliças, 2023. cap. 12.

ONOFRE, R. B. et al. **Nighttime application of UV-C light to control cucurbit powdery mildew**. Plant Health Progress, v. 21, n. 1, p. 31–36, 2020.

PEDREGOSA, F. et al. **Scikit-learn: machine learning in Python**. Journal of Machine Learning Research, v. 12, p. 2825–2830, 2011.

SIDIBÉ, A. et al. **Preharvest UV-C hormesis induces key genes associated with homeostasis, growth and defense in lettuce**. Frontiers in Plant Science, v. 13, 2022.

SMITH, I. M. et al. **Bremia lactucae**. In: CABI COMPENDIUM. Wallingford: CABI, 2021.Disponível em: https://www.cabidigitallibrary.org/doi/10.1079/cabicompendium.10696. Acesso em: mai. 2025.

SUTHAPARAN, A. et al. **Specific alteration of plant-pathogen interactions by UV-B radiation: inactivation of powdery mildew infections by low doses of UV-B**. Journal of Photochemistry and Photobiology B: Biology, v. 114, p. 10–19, 2012.

TAN, M.; LE, Q. V. EfficientNet: rethinking model scaling for convolutional neural networks. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING, 36., 2019. **Proceedings** [...]. [S. l.]: PMLR, 2019. p. 6105–6114.

VEGETABLES BAYER. **Lettuce downy mildew**. [S. l.]: Bayer, [s.d.]. Disponível em: https://www.vegetables.bayer.com/ca/en-ca/resources/agronomic-spotlights/lettuce-downy-mildew.html. Acesso em: 5 maio 2026.

WALLACH, D. et al. Improving the performance of vegetable leaf wetness duration models in greenhouses using decision tree analysis. **Water**, v. 10, n. 7, p. 869, 2018.

YSSP. **Lettuce Detections Dataset: downy mildew class**. Roboflow Universe, 2024. Disponível em: https://universe.roboflow.com/yssp/lettuce-detections. Acesso em: abr. 2026.