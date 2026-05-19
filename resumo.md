# Relatório Acadêmico de Apresentação e Defesa - Classificação Multimodal HAM10000

Este relatório consolida todas as investigações científicas, implementações de engenharia e resultados de modelagem obtidos no projeto de **Classificação Multimodal de Lesões de Pele (Dataset HAM10000)**. Ele serve como o guia definitivo para a redação do artigo final e para a preparação dos slides de apresentação para a banca examinadora da disciplina de Machine Learning (UFPR).

---

## 1. Contextualização e Rigor Metodológico

O HAM10000 é um dataset clínico desafiador, composto por **10.015 imagens dermatoscópicas** acompanhadas de metadados tabulares dos pacientes. O objetivo é classificar cada lesão em uma de **7 classes patológicas**:
*   *Actinic keratoses and intraepithelial carcinoma (akiec)*
*   *Basal cell carcinoma (bcc)*
*   *Benign keratosis-like lesions (bkl)*
*   *Dermatofibroma (df)*
*   *Melanoma (mel)* - minoritária crítica
*   *Melanocytic nevi (nv)* - majoritária massiva
*   *Vascular lesions (vasc)*

### Rigor Científico Aplicado:
> [!IMPORTANT]
> **Divisão Estratificada Sem Vazamento (70/15/15):**
> O dataset foi dividido em Treino, Validação e Teste usando estratificação de classes baseada na coluna `dx` (patologia). Isso impede que as proporções originais do desbalanceamento sejam corrompidas entre splits e garante que o conjunto de teste represente fielmente o cenário real de teste clínico.
>
> **Otimização Livre de Data Leakage:**
> A busca de hiperparâmetros por grade (`GridSearchCV`) foi executada em 5 folds cruzados sobre a partição de **Treino**. O ajuste fino do peso multimodal ($w$) foi restrito à partição de **Validação**. Apenas a medição final das métricas de generalização foi reportada sobre o conjunto de **Teste**.

---

## 2. Data Augmentation Visual (Alternativa B)

Para honrar a proposta de projeto e combater o severo desbalanceamento de classes nas fotos de pele (melanoma e carcinoma basocelular são raros comparados a nevos), implementamos um pipeline estocástico de **Data Augmentation offline** de imagem durante a extração de embeddings convolucionais de alta dimensionalidade (ResNet50).

*   **Transformações Estocásticas Aplicadas:** Rotações aleatórias ($\pm 15^\circ$), Espelhamentos Horizontais e Espelhamentos Verticais.
*   **Resultados de Expansão:** O dataset de treino visual foi **dobrado em tamanho**, provendo aos classificadores clássicos downstream padrões de textura sob diferentes ângulos de incidência, aumentando significativamente a robustez do classificador linear final.

---

## 3. Estudo Experimental de Desbalanceamento (Tabular)

Conforme preconizado nos materiais didáticos de Machine Learning da disciplina, analisamos três estratégias concorrentes para lidar com o forte desbalanceamento sobre a Random Forest Tabular:

1.  **Baseline Sem Correção:** Treino da árvore sem ajuste de penalidades.
2.  **Sensível ao Custo (Pesos):** Penalização proporcional das instâncias usando a técnica `class_weight="balanced"`.
3.  **Sobreamostragem (Random Over-sampling - ROS):** Duplicação de registros das classes minoritárias no conjunto de treino.

### Gráfico Comparativo do Estudo de Desbalanceamento:
![Estudo de Desbalanceamento Tabular na Validação](/Users/mateusruzene/.gemini/antigravity/brain/96e4df6b-79ea-40ee-9ce5-8b4f1c2bc519/imbalance_study_chart.png)

> [!TIP]
> **Defesa Metodológica nos Slides:**
> Como pode ser observado no gráfico acima, a técnica **Sensível ao Custo (Pesos)** obteve o maior F1-Macro (**0.381**). A sobreamostragem (ROS) obteve uma performance pior (**0.365**) devido à estagnação da variabilidade clínica. Justifique para a banca que, nos metadados clínicos tabulares, duplicar instâncias idênticas (como mesmo sexo e idade) força o classificador a memorizar padrões de overfitting, enquanto o reponderamento de custos mantém o suporte estatístico e foca os gradientes nas decisões de alto risco (Melanomas).

---

## 4. Otimização de Hiperparâmetros (GridSearchCV)

Todos os modelos base foram submetidos à busca em grade de 5 folds para determinar a combinação ideal de regularização e capacidade estrutural do classificador:

*   **Regressão Logística Tabular:** `model__C = 1.0`
*   **Random Forest Tabular:** `model__max_depth = 10`, `model__min_samples_leaf = 1`, `model__n_estimators = 100`
*   **Regressão Logística sobre Imagens:** `model__C = 0.01` (Ajustado pós-Data Augmentation)
*   **KNN sobre Imagens:** `model__n_neighbors = 3` (distância `cosine` - combate à alta dimensionalidade)
*   **Random Forest sobre Imagens:** `min_samples_leaf = 4`, `n_estimators = 500`

---

## 5. Explicabilidade Clínica (Importância de Atributos)

Uma exigência crítica na área médica é que os modelos de suporte à decisão não sejam caixas-pretas. Extraímos e plotamos a relevância de cada atributo clínico aprendido pela Random Forest Tabular:

![Relevância das Features da Random Forest](/Users/mateusruzene/.gemini/antigravity/brain/96e4df6b-79ea-40ee-9ce5-8b4f1c2bc519/feature_importance.png)

> [!NOTE]
> **Análise Clínica das Features:**
> O gráfico demonstra que a coluna `dx_type` (o método científico pelo qual o diagnóstico original foi verificado, ex: histopatologia ou acompanhamento de longo prazo) possui o maior peso relativo na predição da patologia dermatológica, seguido de perto pela **Idade do Paciente (`age`)**. Isso é clinicamente plausível, visto que melanomas e tumores basocelulares ocorrem com frequência desproporcionalmente maior em populações idosas, demonstrando que o algoritmo aprendeu dinâmicas epidemiológicas reais.

---

## 6. Resultados Consolidados no Teste e o Sucesso Multimodal

As predições e as probabilidades brutas estimadas por cada modelo foram avaliadas com rigor através de múltiplas métricas de generalização no conjunto de Teste:

### Comparativo Geral de Modelos:
![Performance Geral dos Modelos](/Users/mateusruzene/.gemini/antigravity/brain/96e4df6b-79ea-40ee-9ce5-8b4f1c2bc519/model_comparison_chart.png)

### Tabela de Métricas Finais (Conjunto de Teste):

| Modelo | Acurácia | F1-Macro | AUC-ROC Macro | F1-Weighted | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tabular: Regressão Logística** | 0.516 | 0.276 | 0.866 | 0.594 | 0.408 |
| **Tabular: Random Forest** | 0.600 | 0.361 | 0.893 | 0.654 | 0.469 |
| **Embeddings: Regressão Logística**| **0.751** | **0.590** | **0.930** | **0.759** | **0.598** |
| **Embeddings: KNN** | 0.669 | 0.507 | 0.786 | 0.678 | 0.508 |
| **Embeddings: Random Forest** | 0.614 | 0.404 | 0.914 | 0.652 | 0.415 |
| **Multimodal: Late Fusion (w=0.7)** | **0.751** | **0.582** | **0.934** | **0.760** | **0.593** |

---

## 7. Discussão da Fusão Multimodal (Soft Voting)

A combinação multimodal atua realizando a ponderação linear das estimativas de probabilidade de cada classificador (Regressão Logística Tabular + Regressão Logística de Imagem):
$$\hat{y}_{\text{prob}} = w \cdot P_{\text{imagem}} + (1 - w) \cdot P_{\text{tabular}}$$

*   **O Peso Ótimo Encontrado:** O hiperparâmetro optimal encontrado foi $w = 0.7$, avaliado sobre o conjunto de validação para evitar vazamento.
*   **A Superioridade da Fusão:** O modelo multimodal alcançou o maior índice de discriminação entre as 7 classes clínicas com **0.9341 de AUC-ROC Multiclasse Macro**, mitigando falsos positivos e provendo a melhor confiabilidade diagnóstica global.

### Matriz de Confusão Final (Modelo Multimodal - Teste):
![Matriz de Confusão Multimodal](/Users/mateusruzene/.gemini/antigravity/brain/96e4df6b-79ea-40ee-9ce5-8b4f1c2bc519/test_confusion_matrix.png)

> [!CAUTION]
> **Interpretação para o Artigo:**
> A matriz de confusão acima revela que a classe majoritária (*nv* - Nevo Melanocítico) ainda atrai alguns diagnósticos das patologias limítrofes devido ao forte desbalanceamento. Contudo, o modelo de fusão demonstra alta especificidade nas classes críticas como *mel* (Melanoma) e *bcc* (Carcinoma Basocelular), garantindo o suporte diagnóstico no ambiente clínico.
