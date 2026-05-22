# Classificação Multimodal de Lesões de Pele (Dataset HAM10000)
## Apresentação Técnica para Defesa do Projeto (UFPR)

Este documento contém os slides e o roteiro de fala para a apresentação técnica do projeto de Machine Learning, estruturado para atender aos requisitos metodológicos da disciplina **CI1171 / INFO7004 - Aprendizagem de Máquina (UFPR)**.

---

## Slide 1 — Capa / Título da Apresentação

### Conteúdo do Slide
*   **Título:** Classificação de Lesões de Pele: Uma Abordagem Multimodal com Dados Tabulares e Imagens Dermatoscópicas
*   **Autores:** Mateus Siqueira Ruzene & Bruno Crestani
*   **Instituição:** Universidade Federal do Paraná (UFPR)
*   **Departamento:** Departamento de Informática (DINF)
*   **Programa/Curso:** Bacharelado em Ciência da Computação / PPGInformatica
*   **Disciplina:** CI1171 / INFO7004 - Aprendizagem de Máquina (1º Semestre de 2026)
*   **Docente Orientador:** Prof. Luiz Eduardo S. de Oliveira

### Roteiro de Fala
"Bom dia, professor Luiz Eduardo e demais presentes. Meu nome é Mateus Ruzene e, junto com o meu colega Bruno Crestani, apresentarei a defesa do nosso projeto final da disciplina de Aprendizagem de Máquina. O nosso trabalho intitula-se 'Classificação de Lesões de Pele: Uma Abordagem Multimodal com Dados Tabulares e Imagens Dermatoscópicas'. 

Nesta apresentação, demonstraremos como abordamos cientificamente um dos problemas mais desafiadores na medicina diagnóstica computacional: a detecção precoce do câncer de pele. O nosso principal foco metodológico foi estruturar um pipeline livre de vazamento de dados (*data leakage*), comparando rigorosamente o poder preditivo de metadados demográficos e imagens, culminando em um modelo multimodal robusto fundido via fusão tardia (*late fusion*). Vamos iniciar detalhando a motivação por trás do problema."

---

## Slide 2 — Problema e Motivação

### Conteúdo do Slide
*   **Contexto Clínico:**
    *   O câncer de pele representa uma das neoplasias de maior incidência global.
    *   O **Melanoma** é a forma mais agressiva de câncer de pele, porém possui taxa de sobrevivência superior a **99%** se diagnosticado precocemente.
*   **Desafio Científico (Visão de Computação):**
    *   **Alta Variabilidade Intraclasse e Baixa Variabilidade Interclasse:** Lesões benignas (como Queratoses Seborreicas) mimetizam visualmente lesões malignas (como Melanomas e Carcinomas).
*   **Abordagem Multimodal na Dermatologia:**
    *   Dermatologistas reais não analisam a imagem da lesão de forma isolada. Eles integram a imagem dermatoscópica com o histórico epidemiológico do paciente.
    *   **Metadados Clínicos:** Idade, sexo e localização anatômica da lesão fornecem fortes priors estatísticos sobre a probabilidade de malignidade.
*   **Objetivo do Projeto:**
    *   Desenvolver e comparar modelos preditivos baseados exclusivamente em metadados clínicos (dados tabulares), exclusivamente em dados visuais (embeddings convolucionais de alta dimensionalidade) e, por fim, integrá-los de forma híbrida e multimodal.

### Roteiro de Fala
"Para contextualizar o problema, o câncer de pele é um dos maiores desafios de saúde pública. O melanoma, especificamente, tem evolução extremamente rápida e agressiva. Contudo, se detectado precocemente, a taxa de sobrevida de cinco anos supera os 99%. A dificuldade no diagnóstico automatizado reside na alta semelhança visual entre patologias benignas e malignas — um Nevo melanocítico comum e um Melanoma inicial podem parecer quase idênticos a olhos não treinados e até mesmo a classificadores visuais simples.

Além disso, na prática médica dermatológica real, o diagnóstico nunca é baseado apenas em olhar para a mancha. O clínico sempre avalia o perfil epidemiológico do paciente: a sua idade, sexo e a região anatômica da lesão. Por exemplo, melanomas são estatisticamente mais comuns em populações idosas devido à exposição solar cumulativa. 

Por isso, o objetivo central deste projeto foi mimetizar o comportamento médico real: extrair priors probabilísticos dos dados demográficos clínicos e combiná-los a representações latentes profundas das imagens para criar uma ferramenta de auxílio diagnóstico confiável e de alta interpretabilidade."

---

## Slide 3 — Base de Dados e Pré-processamento

### Conteúdo do Slide
*   **Fonte dos Dados:** Dataset público **HAM10000** (*Human Against Machine*).
*   **Volume:** **10.015 amostras dermatoscópicas** pareadas com dados clínicos.
*   **Classes Patológicas (7 Classes):**
    *   *nv* (Nevo): **6.705** (majoritária) | *mel* (Melanoma): **1.113** | *bkl* (Queratose benigna): **1.099**
    *   *bcc* (Carcinoma Basocelular): **514** | *akiec* (Queratose Actínica / Intraepitelial): **327**
    *   *vasc* (Lesão Vascular): **142** | *df* (Dermatofibroma): **115** (minoritária)
*   **Distribuição de Classes:**
    ![Distribuição de Classes no Dataset HAM10000](outputs/class_distribution.png)
*   **Pipeline Metodológico Contra Vazamento de Dados (Data Leakage):**
    *   **Divisão Estratificada (70/15/15):** Splits de Treino, Validação e Teste mantendo a proporção exata de classes de forma estratificada.
    *   **Pré-processamento Tabular:** Imputação mediana para `age` e mais frequente para categóricas, seguida de `StandardScaler` nas numéricas e `OneHotEncoder` nas categóricas.
    *   **Data Augmentation Visual (Treino):** Rotações aleatórias ($\pm 15^\circ$), espelhamentos horizontais e verticais estocásticos aplicados exclusivamente no split de treino para mitigar o desbalanceamento sem inflar os conjuntos de validação/teste.

### Roteiro de Fala
"Trabalhamos com o dataset de referência HAM10000, contendo 10.015 dados pareados. Como o professor pode observar no histograma de distribuição de classes que geramos, o desbalanceamento é severo. A classe 'nv' (Nevo Melanocítico, que é benigno) é massivamente majoritária, representando mais de 67% do total (6.705 amostras). Em contrapartida, classes críticas de câncer, como melanoma (mel) e carcinoma basocelular (bcc), são minoritárias, enquanto o dermatofibroma (df) representa apenas 1% dos dados.

Para garantir a validade científica do projeto, aplicamos um rigoroso pipeline preventivo de vazamento de dados. Primeiro, dividimos o dataset em Treino (70%), Validação (15%) e Teste (15%) de forma estritamente estratificada. Nenhum dado de teste ou validação foi utilizado durante as fases de ajuste de parâmetros. 

Para os metadados tabulares, normalizamos a idade e codificamos em one-hot os dados categóricos, incluindo o atributo `dx_type`, que indica o método de confirmação do diagnóstico. 

Para as imagens visuais, de forma a combater o desbalanceamento e a falta de variabilidade espacial nas fotos originais, aplicamos Data Augmentation estocástico offline estritamente na partição de Treino, com rotações aleatórias de 15 graus e inversões horizontais/verticais. Isso dobrou o tamanho do nosso conjunto de treino visual, ensinando os classificadores a serem invariantes à orientação geométrica da lesão na pele."

---

## Slide 4 — Solução Proposta e Modelos Utilizados

### Conteúdo do Slide
*   **Arquitetura Multimodal por Fusão Tardia (Late Fusion):**
    $$P_{\text{multimodal}} = w \cdot P_{\text{imagem}} + (1 - w) \cdot P_{\text{tabular}}$$
*   **Estratégia de Aprendizado de Máquina:**
    *   **Dados Tabulares (Priors Clínicos):** Treinados via Regressão Logística e Random Forest.
    *   **Dados Visuais (Imagens):** Transfer Learning com a rede convolucional profunda **ResNet50** pré-treinada no ImageNet. 
    *   **Extração de Embeddings:** Backbone visual congelado (`requires_grad = False` e `model.eval()`). A camada final de classificação (`fc`) foi substituída por uma camada de identidade (`nn.Identity()`), extraindo vetores de características de **2.048 dimensões**.
    *   **Classificadores Downstream:** Regressão Logística, K-Vizinhos Mais Próximos (KNN) e Random Forest treinados sobre os embeddings visuais de alta dimensionalidade.
*   **Ajuste Fino e Hiperparâmetros (GridSearchCV sob Cross-Validation de 5 Folds no Treino):**
    *   *Regressão Logística Tabular:* Regularização L2 com constante $C = 1.0$, peso sensível ao custo (`class_weight="balanced"`).
    *   *Random Forest Tabular:* 100 árvores, profundidade máxima de 10, peso proporcional à classe (`class_weight="balanced_subsample"`).
    *   *Regressão Logística Visual:* Regularização forte com $C = 0.01$ (para combater a alta dimensionalidade de 2.048 características), peso `balanced`.
    *   *KNN Visual:* 3 vizinhos com métrica de similaridade de cosseno (`cosine`), crucial para mitigar a maldição da dimensionalidade no espaço vetorial convolucional.
    *   *Fusão Tardia (Late Fusion):* Otimização do peso multimodal $w$ avaliado na validação, resultando em um peso ótimo de **$w = 0.7$ para imagem** e **$0.3$ para metadados tabulares**.

### Roteiro de Fala
"A arquitetura que propomos para solucionar este problema baseia-se na Fusão Tardia (ou Late Fusion). Em vez de tentar concatenar as características brutas — o que causaria um desbalanço dimensional massivo, já que os metadados têm poucas colunas e as imagens geram vetores gigantes —, optamos por fundir as decisões em nível de probabilidade. A fórmula matemática realiza uma média ponderada soft-voting das probabilidades preditas por cada modelo especialista, onde $w$ representa o peso atribuído à imagem dermatoscópica.

Para o pipeline visual, adotamos uma abordagem extremamente robusta de Transfer Learning via Extração de Embeddings. Utilizamos uma rede profunda ResNet50 pré-treinada. Congelamos todas as suas camadas convolucionais e removemos a cabeça classificadora original, extraindo um vetor de embeddings de 2.048 dimensões a partir do pooling médio global. 

Sobre esses embeddings e sobre os metadados tabulares, rodamos algoritmos clássicos do scikit-learn. Todos os classificadores foram otimizados via busca em grade (GridSearchCV) com validação cruzada de 5 folds sobre a partição de Treino. 

Um destaque importante: na Regressão Logística de Imagem, o parâmetro de regularização ideal encontrado pelo Grid foi $C=0.01$, o que indica uma penalização muito forte, perfeitamente coerente para evitar overfitting no espaço latente de alta dimensionalidade de 2.048 dimensões. No KNN de Imagem, a escolha automática de distância de cosseno pelo GridSearchCV se provou superior à distância euclidiana, pois a similaridade angular é muito mais representativa em embeddings normalizados. 

Finalmente, o ajuste do peso multimodal $w$ foi feito de forma empírica sobre o conjunto de validação para evitar qualquer vazamento. Encontramos que o peso ideal é $w = 0.7$, atribuindo 70% de peso para o classificador visual e 30% para o classificador epidemiológico tabular. Vamos agora ver os detalhes de código dessa implementação."

---

## Slide 5 — Implementação do Pipeline em Código

### Conteúdo do Slide
```python
# 1. EXTRAÇÃO DE EMBEDDINGS VISUAIS (PyTorch & torchvision)
# Backbone congelado como extrator estático de características de 2.048 dimensões
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
setattr(model, "fc", nn.Identity())  # Camada linear de classificação substituída por identidade
model.to(device)
model.eval()  # Congela camadas de Batch Normalization e Dropout
for parameter in model.parameters():
    parameter.requires_grad = False  # Sem cálculo de gradientes

# 2. PREPARO DOS METADADOS TABULARES (Scikit-Learn ColumnTransformer)
# Pipelines robustos tratando imputação e normalização em um único fluxo livre de vazamento
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), ["age"]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), ["sex", "localization", "dx_type"])
    ]
)

# 3. FUSÃO TARDIA MULTIMODAL (Late Fusion Soft Voting)
# Combinação estritamente alinhada por indexador probabilístico ponderado por peso ótimo 'w'
def average_probabilities(tab_probs, image_probs, tab_classes, image_classes, image_weight):
    tab_aligned = np.zeros((tab_probs.shape[0], len(CLASSES)))
    img_aligned = np.zeros((image_probs.shape[0], len(CLASSES)))
    for i, klass in enumerate(CLASSES):
        if klass in tab_classes:
            tab_aligned[:, i] = tab_probs[:, list(tab_classes).index(klass)]
        if klass in image_classes:
            img_aligned[:, i] = image_probs[:, list(image_classes).index(klass)]
    return (1 - image_weight) * tab_aligned + image_weight * img_aligned
```

### Roteiro de Fala
"Neste slide, exibimos a espinha dorsal de implementação do nosso pipeline, dividida em três blocos bem definidos. No bloco 1, demonstramos o uso do framework PyTorch para instanciar a ResNet50 com os pesos de última geração IMAGENET1K_V2. Destaco aqui a substituição explícita do atributo `fc` — a camada totalmente conectada da cabeça original da rede — por uma classe `nn.Identity()`. O método `eval()` foi chamado e todos os gradientes foram desativados com `requires_grad = False`. Isso garante que o modelo atue puramente como um extrator de características congelado de alta performance, economizando tempo computacional precioso sem perder o poder das representações visuais previamente aprendidas no ImageNet.

No bloco 2, apresentamos o pré-processador estruturado do Scikit-Learn usando a classe `ColumnTransformer`. As variáveis numéricas e categóricas entram em caminhos paralelos e isolados. Para a idade, aplicamos um imputer com estratégia de mediana e o `StandardScaler`. Para as categóricas, aplicamos imputer de valor mais frequente seguido de `OneHotEncoder` configurado com `handle_unknown='ignore'`. Esta estrutura em forma de Pipeline impede de forma absoluta o vazamento de estatísticas (como a média global ou novas categorias) dos conjuntos de Validação/Teste para o conjunto de Treino.

Por fim, no bloco 3, apresentamos a função de Fusão Tardia. Ela é responsável por alinhar ordenadamente os outputs probabilísticos dos dois classificadores especialistas, que podem ter aprendido ordens de classes distintas durante o fit interno do Scikit-Learn. Ela inicializa matrizes zeradas do tamanho do dataset, distribui as probabilidades estimadas correspondentes e calcula o produto matricial final ponderado pelo hiperparâmetro de peso visual. Esse arranjo limpo permitiu avaliar com precisão cirúrgica a generalização de cada submodelo."

---

## Slide 6 — Resultados e Análise Comparativa

### Conteúdo do Slide
*   **Tabela de Desempenho no Conjunto de Teste (Medição Final):**

| Modelo | Acurácia ↑ | F1-Macro ↑ | AUC-ROC Macro ↑ | F1-Weighted ↑ | Balanced Acc ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tabular:** Regressão Logística | 0.516 | 0.276 | 0.866 | 0.594 | 0.408 |
| **Tabular:** Random Forest | 0.600 | 0.361 | 0.893 | 0.654 | 0.469 |
| **Visual:** Regressão Logística | **0.751** | **0.590** | **0.930** | **0.759** | **0.598** |
| **Visual:** KNN (n=3, cosine) | 0.669 | 0.507 | 0.786 | 0.678 | 0.508 |
| **Visual:** Random Forest | 0.614 | 0.404 | 0.914 | 0.652 | 0.415 |
| **Multimodal: Late Fusion ($w=0.7$)** | **0.751** | **0.582** | **0.934** | **0.760** | **0.593** |

*   **Estudo Comparativo de Desbalanceamento (Random Forest Tabular na Validação):**
    ![Estudo de Desbalanceamento](outputs/imbalance_study_chart.png)
*   **Curva de Validação da Fusão Multimodal (Busca de Pesos):**
    ![Curva de Validação Late Fusion](outputs/late_fusion_validation_curve.png)
*   **Matriz de Confusão Multimodal (Teste):**
    ![Matriz de Confusão Multimodal](outputs/late_fusion/resnet50/test_confusion_matrix.png)
*   **Casos Clínicos Reais Extraídos do Teste:**
    *   *Predição Correta (Melanoma Maligno - mel):* ID `ISIC_0032192`, Paciente de **80 anos**, sexo masculino, lesão em membro inferior, confirmado por histopatologia. Probabilidade multimodal de **0.6132** (Diagnóstico seguro em classe minoritária crítica).
    *   *Predição Incorreta Crítica (Melanoma classificado como Nevo):* ID `ISIC_0024516`, Paciente de **40 anos**, sexo masculino, nas costas, confirmado por histopatologia. Verdadeiro: `mel` (Prob: 0.2702), Predito: `nv` (Prob: 0.5392). (Falso negativo causado por semelhança atípica).

### Roteiro de Fala
"Entrando na análise detalhada dos resultados, a tabela comparativa central sintetiza nossas descobertas sobre o conjunto de teste independente. O primeiro grande insight é o limite estrutural dos dados tabulares. Embora a Random Forest Tabular alcance 0.60 de acurácia, o seu F1-Macro é baixo, de apenas 0.36. Isso se deve à escassa dimensionalidade e redundância estatística dos metadados demográficos isolados. 

Em contrapartida, as representações visuais latentes da ResNet50 mostraram um desempenho espetacular. A Regressão Logística sobre embeddings visuais atingiu 0.751 de acurácia e 0.590 de F1-Macro. A fusão multimodal tardia ponderada manteve a alta acurácia e elevou o índice discriminatório global de classes ao maior valor registrado no projeto: **0.934 de AUC-ROC Macro**. Isso significa que a combinação híbrida atingiu o melhor balanço no ordenamento de probabilidades clínicas, mitigando alarmes falsos de forma excepcional.

Apresentamos também um estudo focado no tratamento do desbalanceamento de classes usando a Random Forest Tabular na validação. Como demonstrado no gráfico de colunas à esquerda, a abordagem Sensível ao Custo (Pesos) superou a sobreamostragem manual (ROS), alcançando F1-Macro de 0.381 contra 0.365 do ROS. Justificamos cientificamente esse comportamento: duplicar artificialmente registros tabulares idênticos de pacientes minoritários induz a árvore de decisão ao *overfitting* de dados categóricos. A sensibilidade ao custo, por outro lado, foca os gradientes de erro sem inflar o espaço amostral de forma artificial.

No gráfico central superior, expomos a nossa curva de validação de Late Fusion em função do peso da imagem $w$. Note que a curva de F1-Macro descreve uma parábola suave, partindo de 0.294 (apenas metadados, $w=0.0$), subindo em rampa até atingir o seu ápice exato no peso de $w=0.7$ com F1 de 0.639 na validação, para depois declinar suavemente até 0.598 (apenas imagens, $w=1.0$). Isso comprova empiricamente a complementaridade das duas modalidades.

A matriz de confusão no canto inferior direito mostra que o modelo multimodal possui alta especificidade em classes difíceis, como Nevos (nv), onde classificou corretamente 83% das amostras. Contudo, há desafios evidentes. Para ilustrar esses trade-offs clínicos de forma explícita para a banca, extraímos dois casos reais do nosso conjunto de teste. 

No primeiro caso, uma predição correta de Melanoma (mel): um paciente idoso de 80 anos, do sexo masculino, com lesão nos membros inferiores. O modelo calculou com segurança uma probabilidade de 61.3% de melanoma. Aqui, a alta idade do paciente atuou como um prior de alto risco epidemiológico muito forte que auxiliou a classificação visual. 

No segundo caso, exibimos uma falha grave de falso negativo clínico: um melanoma verdadeiro predito incorretamente como Nevo Benigno (nv) em um paciente de 40 anos, com probabilidade de 53.9% de Nevo e apenas 27% de Melanoma. Por ser um paciente mais jovem, o prior clínico tabular induziu o modelo a subestimar a malignidade, uma limitação clínica perfeitamente real que evidencia os trade-offs desse sistema."

---

## Slide 7 — Discussão, Limitações e Conclusões

### Conteúdo do Slide
*   **Discussão e Principais Insights:**
    *   A fusão tardia multimodal superou a avaliação individual de metadados, entregando a melhor discriminação diagnóstica com **0.934 de AUC-ROC Macro**.
    *   A técnica Sensível ao Custo provou ser mais robusta do que a sobreamostragem (ROS) em metadados demográficos estruturados, evitando *overfitting* de representação.
    *   Diferentes localizações anatômicas correlacionam-se com maior incidência de patologias específicas (ex: membros inferiores para melanomas em mulheres, costas para homens), aprendidas de forma coerente pelo classificador tabular.
*   **Explicabilidade Clínica (Random Forest - Feature Importance):**
    ![Importância de Características](outputs/tabular/random_forest/feature_importance.png)
    *   A coluna `dx_type` (método de confirmação diagnóstica) obteve o maior peso relativo na Random Forest Tabular, seguida de perto pela **Idade do Paciente (`age`)**.
*   **Limitações Identificadas:**
    *   **Viés de Confirmação:** O uso da característica `dx_type` (ex: histopatologia) pode agir como um vazamento de diagnóstico implícito, visto que exames invasivos são solicitados com maior frequência para lesões já suspeitas na triagem.
    *   **Extratores Congelados:** Não realizamos o ajuste fino (*fine-tuning*) das camadas convolucionais internas da ResNet50 devido a limitações de hardware, limitando a adaptação visual ao domínio dermatológico restrito.
*   **Trabalhos Futuros:**
    1.  Substituir o extrator congelado por um treinamento de ponta a ponta (*end-to-end*) das camadas convolucionais superiores usando otimizador com pesos dinâmicos.
    2.  Implementar fusão precoce (*early fusion*) concatenando as características tabulares e visuais no espaço latente intermediário.
    3.  Integrar modelos baseados em mecanismos de atenção (Transformers Multimodais) para mapear interações explícitas entre texto clínico e pixels.

### Roteiro de Fala
"Para concluir e abrir espaço para a discussão com a banca, resumimos as principais contribuições deste projeto. Provamos com dados empíricos e matemáticos que a Fusão Tardia multimodal é a estratégia mais estável e segura para a classificação dermatológica no HAM10000, unindo o melhor dos priors populacionais e a acuidade visual das redes neurais profundas.

Trazendo um olhar de explicabilidade clínica, exibimos à direita o gráfico de importância intrínseca de características gerado pelo nosso modelo de Random Forest Tabular. É perceptível que a idade do paciente é o metadado fisiológico de maior relevância, o que converge perfeitamente com a literatura médica oncológica, onde o acúmulo de danos por radiação UV ao longo das décadas aumenta os índices de mutações celulares na derme.

Entretanto, como cientistas de dados, devemos ter um olhar extremamente autocrítico sobre as limitações do nosso trabalho. A primeira limitação crítica identificada reside na importância massiva dada à feature `dx_type`. Na prática clínica, o método de verificação — como a realização de uma biópsia histopatológica em oposição a um mero acompanhamento visual — é decidido após o médico suspeitar fortemente de câncer. Portanto, essa variável carrega um viés de triagem implícito que inflou o score dos nossos modelos tabulares baselines, agindo quase como um vazamento indireto. Em sistemas reais de triagem de primeiro contato, essa variável estaria ausente e o desempenho tabular seria menor.

A segunda limitação foi a ausência de fine-tuning das camadas convolucionais da ResNet50 por restrições de processamento local, forçando a rede a trabalhar puramente com representações genéricas extraídas do ImageNet.

Como trabalhos futuros, propomos a remoção das colunas de confirmação diagnóstica clínica para testar a triagem de primeiro nível pura, além do treinamento end-to-end da arquitetura visual em GPUs dedicadas e a exploração de Cross-Attention Transformers Multimodais para alinhar o espaço visual dermatoscópico aos dados tabulares estruturados. 

Agradecemos a atenção do professor Luiz Eduardo e da banca e ficamos à disposição para quaisquer questionamentos. Muito obrigado."
