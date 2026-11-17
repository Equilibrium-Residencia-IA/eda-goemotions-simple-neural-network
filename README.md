# Classificação de emoções em português com rede neural em NumPy

Classificador **multirrótulo** que identifica as **7 emoções de Ekman** (medo, tristeza, raiva, nojo, surpresa,
alegria e neutro) em mensagens em português. A rede neural foi **implementada do zero em NumPy**, sem PyTorch ou
TensorFlow, e usa features **TF-IDF**.

Os dados são a [versão em português do GoEmotions](https://github.com/Equilibrium-Residencia-IA/emotional-dataset-pt-br):
cerca de 54 mil comentários do Reddit traduzidos, com 28 rótulos de emoção.

## Conteúdo do repositório

| Arquivo | O que faz |
|---|---|
| `eda_goemotions_ptbr.ipynb` | Baixa o dataset, trata os textos, une train/dev/test em um dataset só, faz a análise exploratória e gera os arquivos de treino |
| `rede_neural_tfidf_ekman.ipynb` | Extrai as features TF-IDF, treina a rede neural, ajusta os limiares, avalia no teste e compara com uma regressão logística |
| `requirements.txt` | Pacotes Python necessários |

Os notebooks devem ser executados **nessa ordem**: o primeiro gera os CSVs que o segundo usa.

```
eda_goemotions_ptbr.ipynb ──► data_processed/train.csv
                              data_processed/validation.csv  ──► rede_neural_tfidf_ekman.ipynb
                              data_processed/test.csv
```

## Como rodar

### Opção 1: no computador

Requisitos: Python 3.10 ou mais recente e `git` instalado (o primeiro notebook usa o `git` para baixar o dataset).

```bash
git clone <url-deste-repositorio>
cd <nome-deste-repositorio>

python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook
```

No Jupyter:

1. Abra `eda_goemotions_ptbr.ipynb` e execute tudo (*Run → Run All Cells*). Ele clona o repositório do dataset
   em `emotional-dataset-pt-br/` e salva os resultados em `data_processed/`. Leva cerca de 2 minutos.
2. Abra `rede_neural_tfidf_ekman.ipynb` e execute tudo. Ele lê os CSVs de `data_processed/` e treina a rede.
   Leva menos de 2 minutos em CPU.

### Opção 2: no Google Colab

1. Abra `eda_goemotions_ptbr.ipynb` no Colab e execute tudo. Ao terminar, baixe `train.csv`, `validation.csv`
   e `test.csv` da pasta `data_processed/`, pelo painel de arquivos à esquerda.
2. Abra `rede_neural_tfidf_ekman.ipynb` no Colab e execute tudo. Quando os CSVs não estão na pasta, o notebook
   pede o upload: envie os três arquivos do passo anterior.

## Configurações

Cada notebook tem uma **célula de configuração** no início, onde ficam as decisões ajustáveis.

- **EDA:** número mínimo de palavras por texto, se o rótulo `neutral` sai quando aparece junto com outra emoção, e
  se os comentários que são só `neutral` são descartados.
- **Rede neural:** número de features TF-IDF (`N_FEATURES`), camadas ocultas (`HIDDEN`), função de ativação, taxa de
  aprendizado, épocas, tamanho do lote e semente aleatória.

## Resultados

Medidos no conjunto de teste (5.357 mensagens):

| Métrica | Rede neural | Regressão logística |
|---|---|---|
| ROC-AUC macro | 0,864 | 0,863 |
| F1 macro | 0,542 | 0,547 |
| F1 micro | 0,613 | 0,609 |
| Acurácia da emoção principal | 0,593 | — |

A rede e a regressão logística ficam praticamente empatadas: com TF-IDF, as features fazem a maior parte do trabalho.
Alegria é a emoção mais fácil (F1 0,75). Raiva e nojo são as mais difíceis (F1 0,45 e 0,42), e o erro mais comum é
classificar uma emoção como neutro.

**Variação entre execuções:** os números podem mudar um pouco conforme a semente aleatória e o ambiente. Por exemplo,
Colab e computador local usam bibliotecas de álgebra linear diferentes. Em 10 sementes, o F1 macro ficou em
**0,552 ± 0,004** e a ROC-AUC macro em **0,864 ± 0,001**. Na mesma máquina e com a mesma semente, o resultado se repete
exatamente.

## Observações

- Os exemplos de texto exibidos no notebook de EDA passam por um filtro que esconde comentários com conteúdo sexual,
  palavrões pesados ou ofensas. O dataset em si não é alterado, e esse tipo de conteúdo continua nos dados de treino.
- Os rótulos têm ruído herdado do GoEmotions original, e o conteúdo vem de discussões do Reddit em inglês traduzidas,
  com muitos temas dos EUA.
- As pastas `emotional-dataset-pt-br/` e `data_processed/` são geradas pelos notebooks e ficam fora do repositório
  (veja o `.gitignore`).
