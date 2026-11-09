# TCC: Transcricao Automatica de Audio de Guitarra para Tablatura

## Escopo e status atual

O escopo experimental atual e **monofonico**: cada intervalo contem no
maximo uma nota. A transcricao principal trabalha no sinal original, sem usar
a CQT ou morfologia para decidir onsets e f0:

- [x] Onsets pelo fluxo espectral (`librosa.onset.onset_detect`).
- [x] f0 global pelo pYIN, agregado nos intervalos entre onsets.
- [x] Avaliacao quantitativa com benchmark de notas monofonicas isoladas (IDMT-SMT-GUITAR): 100% de acuracia de pitch em notas isoladas.
- [x] CQT e mascara preservadas como analise exploratoria documentada.
- [x] Mapeamento de tablatura (resolucao de ambiguidade corda/casa via menor esforco biomecanico).
- [x] Exportacao e visualizacao da tablatura em texto formatado (ASCII Tab).
- [ ] Mapeamento avancado de digitação para passagens polifônicas complexas (Trabalhos Futuros com GuitarSet).

> **Alinhamento Metodológico:** Para o escopo monofônico adotado no TCC 1, o benchmark oficial é o **IDMT-SMT-GUITAR (Fraunhofer IDMT)**, que fornece notas isoladas limpas com ground truth de corda e casa. O GuitarSet fica catalogado como referência para trabalhos futuros de expansão polifônica.

## Estrutura do repositorio

* `audios/`: arquivos locais de teste ativos.
* `dataset/IDMT-SMT-GUITAR_V2/`: dataset de referência (ignorado no Git pelo `.gitignore`).
* `historico_etapa1_e_etapa2/`: arquivo histórico dos áudios exploratórios originais (`som1` a `som5`) e das figuras geradas nas fases anteriores para fins de documentação do TCC.
* `resultados/`: figuras e relatórios quantitativos em CSV gerados pelas pipelines.
* `pipeline_transcritor.py`: pipeline monofônica (onset + pYIN).
* `avaliar_com_idmt.py`: script de avaliação quantitativa com o dataset IDMT-SMT-GUITAR.
* `avaliar_com_guitarset.py`: avaliação JAMS/GuitarSet com `mir_eval`.
* `analise_exploratoria.py`: figuras para o capítulo de análise exploratória.
* `gerar_espectrograma.py`: visualização da máscara CQT para fins exploratórios.
* `RELATORIO_TECNICO.md`: fundamentação teórica, limites e protocolo experimental.

## Como executar

### 1. Dependencias

```bash
pip install -r requirements.txt
```

### 2. Download do Dataset de Benchmark (IDMT-SMT-GUITAR)

O dataset utilizado para testes com notas isoladas limpas é o **IDMT-SMT-GUITAR** (Fraunhofer IDMT). Como seu tamanho total é grande, ele está configurado no `.gitignore`.

* **Download oficial (Zenodo):** [https://zenodo.org/records/7544110](https://zenodo.org/records/7544110)
* Após baixar o arquivo compactado (`IDMT-SMT-GUITAR_V2.zip`), descompacte-o na pasta `dataset/` para que o caminho fique:
  `dataset/IDMT-SMT-GUITAR_V2/`

### 3. Executando o Benchmark Automatizado com o IDMT

Para avaliar automaticamente as notas isoladas e confrontar com o gabarito XML oficial gerando relatório CSV:

```bash
python avaliar_com_idmt.py --max 20
```

### 4. Transcricao Monofonica Manual

Para testar um arquivo individual e ver a tabela de notas com onsets e offsets:

```bash
python pipeline_transcritor.py "dataset/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC/audio/G53-40100-1111-00001.wav" --visualizar
```

