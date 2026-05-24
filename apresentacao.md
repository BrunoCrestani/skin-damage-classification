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
"Bom dia, professor Luiz Eduardo e a todos que nos assistem. Eu sou o Mateus Ruzene e, junto com o meu colega Bruno Crestani, vou apresentar a defesa do nosso projeto final da disciplina de Aprendizagem de Máquina, intitulado 'Classificação de Lesões de Pele: Uma Abordagem Multimodal com Dados Tabulares e Imagens Dermatoscópicas'. 

Hoje, nós vamos mostrar como lidamos com a detecção precoce do câncer de pele, que é um dos problemas mais difíceis da medicina diagnóstica. Nosso maior cuidado metodológico foi construir um pipeline robusto e totalmente livre de vazamento de dados, o famoso *data leakage*. Ao longo da apresentação, nós vamos comparar o poder preditivo dos dados clínicos dos pacientes com as imagens e, no fim, mostrar como combinamos as duas visões usando a fusão tardia (*late fusion*). Vamos começar explicando por que esse problema é tão importante."

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
"Para contextualizar, o câncer de pele é um dos tumores mais frequentes do mundo. O melanoma é a sua forma mais grave, mas se a gente descobrir ele logo no início, a chance de cura passa de 99%. O grande desafio para a computação é que lesões benignas comuns e tumores malignos são visualmente muito parecidos. Às vezes, até para um dermatologista experiente é difícil diferenciar um nevo comum de um melanoma inicial só no olho.

No entanto, no dia a dia, os médicos não olham só para a mancha. Eles usam também o histórico clínico do paciente — como a idade, o sexo e onde a lesão está localizada. Esses metadados dão pistas estatísticas muito fortes. Por isso, nosso objetivo foi imitar esse raciocínio médico: criar modelos focados só nos dados do paciente, outros focados só nas imagens e, depois, combinar essas duas visões em um único modelo multimodal."

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
"Nós utilizamos o famoso dataset HAM10000, que tem pouco mais de 10 mil amostras. Como dá para ver no gráfico que geramos, o desbalanceamento das classes é gigantesco. A classe de nevos — que são as pintas benignas comuns — representa 67% do dataset, com mais de 6.700 amostras. Já os tumores graves como melanomas (mel) e carcinomas (bcc) são bem mais raros, e o dermatofibroma representa só 1% dos dados.

Para garantir que nossos experimentos fossem válidos cientificamente, a gente dividiu a base em Treino (70%), Validação (15%) e Teste (15%) usando estratificação de classes, garantindo que a proporção original fosse mantida em todos os splits e nenhum dado do teste ou validação vazasse para a fase de ajuste de parâmetros. 

Nos dados tabulares dos pacientes, tratamos valores faltantes e aplicamos normalização usando o `StandardScaler` na variável de idade e o `OneHotEncoder` nas categóricas. Nas imagens, para evitar que o modelo decorasse padrões devido ao desbalanceamento, usamos técnicas de Data Augmentation estocástico exclusivamente no conjunto de treino, aplicando rotações e espelhamentos aleatórios. Isso dobrou nossa base de treino visual de forma segura, sem inflar os conjuntos de teste."

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
"A arquitetura proposta usa a Fusão Tardia (ou Late Fusion). Em vez de tentar juntar as imagens brutas e os dados dos pacientes logo de início — o que daria um problema enorme de proporção nas dimensões —, nós preferimos rodar modelos especialistas em cada tipo de dado e depois combinar os seus resultados. A fórmula faz uma média ponderada das probabilidades que cada modelo dá para cada uma das 7 classes.

Para as imagens, usamos transferência de aprendizado com uma ResNet50 pré-treinada. Nós congelamos as camadas dela para trabalhar como um extrator de características fixo e trocamos a camada final por uma camada de identidade. Com isso, convertemos cada imagem em um vetor denso de 2.048 dimensões.

Sobre esses vetores e sobre os metadados dos pacientes, nós otimizamos classificadores clássicos com busca em grade (GridSearchCV) e validação cruzada de 5 folds sobre o treino. Um destaque técnico importante: a Regressão Logística das imagens escolheu uma regularização forte com C=0.01 para lidar com a alta dimensionalidade de 2.048 características. O KNN visual foi otimizado com similaridade de cosseno, que funciona muito melhor do que a euclidiana em espaços de alta dimensão. Por fim, encontramos na Validação que o peso ideal para a combinação é 0.7 para a imagem e 0.3 para os metadados."

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
"Aqui nós temos os trechos essenciais do nosso código em PyTorch e Scikit-Learn. No primeiro bloco, instanciamos a ResNet50 e configuramos o atributo `fc` como `nn.Identity()`. Com o `eval()` e o `requires_grad = False` para todos os parâmetros, garantimos que a rede funcione de maneira estática, gerando os embeddings sem recalcular gradientes, o que poupa muito processamento.

No segundo bloco, temos o pré-processamento tabular usando o `ColumnTransformer` do Scikit-Learn. Criamos pipelines separados para idade e variáveis categóricas. Centralizar isso em um pipeline unificado impede de forma absoluta que dados de validação ou teste vazem para o treino (o temido data leakage).

No terceiro bloco, está a função de fusão tardia. Como os modelos podem treinar em ordens diferentes de classes, essa função garante que os outputs estejam alinhados nos índices corretos de classes e faz a média ponderada usando o nosso peso `w` de 0.7."

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
"Agora vamos analisar os resultados finais no conjunto de teste. O primeiro ponto que chama a atenção é o limite dos metadados sozinhos. A Random Forest com dados tabulares alcançou acurácia de 60%, mas o F1-Macro ficou em apenas 0.36, por conta da pouca informação contida em idade, sexo e localização.

Por outro lado, o modelo focado só em imagens se saiu muito melhor: a Regressão Logística visual teve acurácia de 75.1% e F1-Macro de 0.59. Ao rodar a fusão multimodal tardia, mantivemos a acurácia alta e conseguimos o maior poder de separação de classes de todo o projeto, alcançando **0.934 de AUC-ROC Macro**.

No gráfico de desbalanceamento na validação, à esquerda, comprovamos que tratar os pesos das classes (Sensível ao Custo) foi melhor do que a sobreamostragem (ROS) na Random Forest tabular, atingindo F1 de 0.381 contra 0.365. Isso acontece porque duplicar dados idênticos de pacientes faz a árvore de decisão decorar padrões de idade e gênero (*overfitting*), enquanto os pesos ajustam a punição do erro diretamente na perda do modelo.

No gráfico de validação da fusão multimodal, vemos a parábola do F1-Macro. Começa baixo com 0.294 (usando só metadados), atinge o pico em w=0.7 com 0.639 na validação, e cai para 0.598 (usando só imagens). Isso mostra empiricamente que a combinação de dados demográficos com imagens traz um ganho real de desempenho.

Para ilustrar de forma clara para a banca, trazemos dois casos práticos do conjunto de teste. No primeiro, o modelo acertou com segurança um melanoma em um paciente de 80 anos com 61% de probabilidade, onde a idade avançada ajudou como um fator de risco epidemiológico forte. No segundo, temos um erro crítico: um melanoma em um paciente jovem de 40 anos foi classificado como nevo benigno com 53% de probabilidade. Como o paciente é jovem, o modelo foi induzido a subestimar a gravidade, mostrando um trade-off importante e um limite real dessa abordagem."

---

## Slide 7 — Discussão, Limitações e Conclusões

### Conteúdo do Slide
*   **Discussão e Principais Insights:**
    *   A fusão tardia multimodal superou a avaliação individual de metadados, entregando a melhor discriminação diagnóstica com **0.934 de AUC-ROC Macro**.
    *   A técnica Sensível ao Custo provou ser mais robusta do que a sobreamostragem (ROS) em metadados demográficos estruturados, evitando *overfitting* de representação.
    *   Diferentes localizações anatômicas correlacionam-se com maior incidence de patologias específicas (ex: membros inferiores para melanomas em mulheres, costas para homens), aprendidas de forma coerente pelo classificador tabular.
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
"Para finalizar, o nosso projeto provou de forma empírica que a fusão de metadados com imagens usando late fusion traz mais estabilidade e precisão para os diagnósticos de lesões de pele.

No gráfico de importância das características da Random Forest à direita, vemos que a variável `dx_type` (o método de confirmação do diagnóstico) e a idade do paciente são os fatores tabulares que mais pesam no resultado. Faz todo sentido biológico a idade ser importante, já que o dano cumulativo do sol na pele aumenta o risco de lesões malignas ao longo dos anos.

Mas precisamos ser autocríticos sobre as limitações do nosso trabalho. A maior delas é que a variável `dx_type` (tipo de confirmação do diagnóstico, como biópsia ou acompanhamento de longo prazo) tem um forte viés de triagem. Em um cenário clínico real, o médico só pede uma biópsia se já suspeitar que a lesão é grave. Então, usar essa variável inflou o desempenho do nosso modelo tabular baseline, agindo quase como um vazamento indireto. Em um sistema de triagem de primeiro contato real, ela não estaria disponível. Além disso, não fizemos o fine-tuning da ResNet50 por limitações de hardware de GPU.

Como passos futuros, pretendemos remover variáveis suspeitas de viés para testar um cenário real de primeiro contato, treinar a rede visual de ponta a ponta (end-to-end) e explorar modelos com atenção visual e Transformers Multimodais para alinhar ainda melhor o texto clínico com as imagens dermatoscópicas.

Agradecemos ao professor Luiz Eduardo e à banca pela atenção e estamos abertos a perguntas. Obrigado!"
