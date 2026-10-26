# TCC: Transcricao Automatica de Audio de Guitarra para Tablatura

## Escopo e status atual

O escopo experimental atual e **monofonico**: cada intervalo contem no
maximo uma nota. A transcricao principal trabalha no sinal original, sem usar
a CQT ou morfologia para decidir onsets e f0:

- [x] Onsets pelo fluxo espectral (`librosa.onset.onset_detect`).
- [x] f0 global pelo pYIN, agregado nos intervalos entre onsets.
- [x] Avaliacao padronizada com anotacoes JAMS do GuitarSet e `mir_eval`.
- [x] CQT e mascara preservadas como analise exploratoria, nao como nucleo
  da transcricao.
- [ ] Validar o lote final do GuitarSet e reportar os resultados quantitativos.
- [ ] Mapeamento de tablatura (ambiguidade de corda/casa).
- [ ] Exportacao para ASCII e MusicXML.

> **Limitacao metodologica:** `-40 dB` e os tamanhos de abertura sao
> configuracoes exploratorias. Nao sao apresentados como parametros
> "calibrados" ate que os experimentos e metricas do GuitarSet sejam
> executados. O fechamento horizontal foi removido do fluxo padrao porque
> pode fundir ataques consecutivos.

## Estrutura do repositorio

* `audios/`: arquivos locais de teste.
* `datasets/guitarset/`: anotacoes e audios do GuitarSet (nao versionados).
* `resultados/`: figuras e relatorios gerados.
* `pipeline_transcritor.py`: pipeline monofonica (onset + pYIN).
* `avaliar_com_guitarset.py`: avaliacao JAMS/GuitarSet com `mir_eval`.
* `analise_exploratoria.py`: figuras para o capitulo de analise exploratoria.
* `gerar_espectrograma.py`: visualizacao legada da mascara CQT, sem fechamento
  por padrao.
* `experimentos_morfologia.py`: tabela descritiva para comparar configuracoes.
* `RELATORIO_TECNICO.md`: fundamentacao, limites e protocolo experimental.

## Como executar

### Dependencias

```bash
pip install -r requirements.txt
```

### Transcricao monofonica

```bash
python pipeline_transcritor.py audios/som1.wav --visualizar
```

### Analise exploratoria

```bash
python analise_exploratoria.py audios/som1.wav
```

### GuitarSet e metricas

Baixe as anotacoes e os audios do
[GuitarSet no Zenodo](https://zenodo.org/records/3371780) e coloque-os em
`datasets/guitarset/annotation` e `datasets/guitarset/audio_mono-mic`.
O modo `solo` e o recomendado para o escopo monofonico:

```bash
python avaliar_com_guitarset.py --batch --style solo --max-files 10 --salvar-csv resultados/guitarset.csv
```

Para registrar comparacoes exploratorias de limiar e abertura:

```bash
python experimentos_morfologia.py audios/som1.wav
```
