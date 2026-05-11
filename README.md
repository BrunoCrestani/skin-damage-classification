# Projeto HAM10000 - Classificacao de Lesoes de Pele

Este diretorio inicia o desenvolvimento do projeto descrito na proposta:
classificar lesoes de pele no dataset HAM10000 usando metadados tabulares,
imagens dermatoscopicas e, em seguida, combinacao multimodal.

## Estrutura

```text
projeto/
  data/
    raw/ham10000/        # coloque aqui os arquivos baixados do Kaggle/ISIC
    processed/           # splits e artefatos intermediarios
  outputs/               # metricas, matrizes de confusao e modelos
  scripts/               # comandos de preparo e treino
  src/skin_lesions/      # codigo reutilizavel
```

## Dataset esperado

Baixe o HAM10000 e coloque os arquivos em `projeto/data/raw/ham10000`.
O codigo procura por:

- `HAM10000_metadata.csv`
- imagens `.jpg` em qualquer subdiretorios, incluindo os nomes comuns
  `HAM10000_images_part_1` e `HAM10000_images_part_2`

## Instalacao

```bash
cd projeto
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Primeiro fluxo experimental

1. Criar splits estratificados:

```bash
python scripts/prepare_ham10000.py
```

2. Treinar baseline tabular:

```bash
python scripts/train_tabular_baseline.py
```

3. Extrair embeddings das imagens com uma CNN pre-treinada:

```bash
python scripts/extract_image_embeddings.py --model resnet50
```

4. Treinar baseline sobre embeddings de imagem:

```bash
python scripts/train_image_baseline.py --model resnet50
```

5. Combinar probabilidades tabulares e de imagem:

```bash
python scripts/train_late_fusion.py --model resnet50
```

Os resultados sao salvos em `outputs/`.

