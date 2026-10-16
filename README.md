# TCC: Transcrição Automática de Áudio de Guitarra para Tablatura

Este repositório contém o código-fonte e os experimentos do meu Trabalho de Conclusão de Curso (TCC). O objetivo principal do projeto é desenvolver um sistema capaz de processar um arquivo de áudio (`.wav`) contendo gravações de guitarra e transcrevê-lo automaticamente para o formato de tablatura.

Para atingir este objetivo, o projeto aborda desafios clássicos de Recuperação de Informação Musical (MIR), como a **estimativa de pitch** e a **resolução da ambiguidade do braço da guitarra** (mesma nota em cordas diferentes).

---

## Status Atual (Progresso)

O projeto concluiu as duas primeiras etapas do pipeline de transcrição:

**Concluído (Etapa 1 - Limpeza Espectral e Morfologia):**
- [x] **Transformada CQT:** Constant-Q Transform calibrada para a afinação da guitarra (fmin = 82.41 Hz / E2, n_bins = 84, hop_length = 512).
- [x] **Limiarização (Noise Gate):** Corte em -40 dB eliminando ruído elétrico e eco ambiente.
- [x] **Máscara Binária:** Separação do sinal musical do ruído através de matrizes booleanas.
- [x] **Morfologia Direcional:** Abertura horizontal (1x5) para desacoplamento de transientes verticais da palheta e fechamento horizontal (1x15) para união de vibrato e sustentação.

**Concluído (Etapa 2 - Extração Musical via Visão Computacional):**
- [x] **Rotulação de Componentes Conexos:** Identificação de objetos conexos e Bounding Boxes temporais/frequenciais (`scipy.ndimage`).
- [x] **Detecção de Onset e Offset:** Extração precisa do tempo de início, término e duração real de cada palhetada.
- [x] **Agrupamento Harmônico (Harmonic Sieve):** Resolução do problema da série harmônica, identificando a Fundamental (f0) e agrupando sobretons concomitantes.
- [x] **Anotação Visual:** Desenho de Bounding Boxes e rótulos das notas identificadas no espectrograma final.

**Próximos Passos (Etapa 3):**
- [ ] **Mapeamento de Tablatura:** Resolução da ambiguidade do braço da guitarra (Fretboard Ambiguity) via otimização biomecânica (HMM / Viterbi).
- [ ] **Exportação:** Geração de tablatura legível em ASCII e MusicXML.

> 📖 **Documentação Acadêmica:** Consulte o arquivo [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md) para a fundamentação matemática, justificativas de cada parâmetro calibrado e referências bibliográficas completas para o TCC.

---

## Estrutura do Repositório

* `audios/`: Arquivos de áudio (.wav / .mp3) com execuções de teste na guitarra.
* `resultados/`: Imagens geradas com os 4 painéis da pipeline (CQT original, máscara bruta, máscara polida e notas detectadas com Bounding Boxes).
* `gerar_espectrograma.py`: Script principal das Etapas 1 e 2 (Pré-processamento, Morfologia e Extração de Notas).
* `RELATORIO_TECNICO.md`: Relatório acadêmico completo com a teoria, decisões de projeto e dados empíricos.

---

## Como executar o projeto

### 1. Dependências
O projeto é desenvolvido em **Python** e requer as bibliotecas matemáticas e de processamento de áudio padrão da indústria. Para instalar todas as dependências, execute no terminal:

```bash
pip install numpy scipy matplotlib librosa
```

### 2. Executando a Limpeza Espectral
Basta rodar o script principal. Ele lerá um arquivo `.wav` da pasta de áudios e gerará a imagem comparativa da aplicação da máscara morfológica.

```bash
python gerar_espectrograma.py
```

O resultado será salvo automaticamente na pasta `resultados/` com um número sequencial (ex: `resultado_1.png`).