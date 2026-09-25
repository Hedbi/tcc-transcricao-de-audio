# TCC: Transcrição Automática de Áudio de Guitarra para Tablatura

Este repositório contém o código-fonte e os experimentos do meu Trabalho de Conclusão de Curso (TCC). O objetivo principal do projeto é desenvolver um sistema capaz de processar um arquivo de áudio (`.wav`) contendo gravações de guitarra e transcrevê-lo automaticamente para o formato de tablatura.

Para atingir este objetivo, o projeto aborda desafios clássicos de Recuperação de Informação Musical (MIR), como a **estimativa de pitch** e a **resolução da ambiguidade do braço da guitarra** (mesma nota em cordas diferentes).

---

## Status Atual (Progresso)

O projeto está na fase de **Pré-processamento e Limpeza Espectral**, preparando a representação visual/matemática do áudio para a extração de características.

**Concluído (Etapa 1 - Limpeza Espectral):**
- [x] **Transformada CQT:** Implementação da Constant-Q Transform calibrada para a afinação padrão da guitarra (fmin = 82.41 Hz / E2).
- [x] **Limiarização (Noise Gate):** Corte de ruído de fundo e reverberação abaixo de -40 dB.
- [x] **Máscara Binária:** Separação do sinal musical do ruído através de matrizes booleanas.
- [x] **Morfologia Matemática:** Uso de operações de Fechamento (1x20) e Abertura (3x3) espaciais para tapar descontinuidades nas notas e remover ilhas isoladas de ruído, transformando a energia acústica em blocos conexos sólidos.

**Próximos Passos (Etapas 2 e 3):**
- [ ] **Detecção de Onset:** Descobrir o tempo exato (ataque) em que cada bloco conexo inicia.
- [ ] **Detecção da Fundamental (Pitch Tracking):** Identificar a frequência fundamental (f0) de cada bloco, ignorando os harmônicos superiores.
- [ ] **Mapeamento de Tablatura:** Modelagem (via HMM ou teoria dos grafos) para definir em qual corda e casa a nota fundamental foi tocada, simulando a biomecânica da mão humana.

---

## Estrutura do Repositório

* `audios/`: Diretório contendo os arquivos `.wav` de teste (gravações de guitarra isolada).
* `resultados/`: Diretório onde o algoritmo salva as imagens geradas (gráficos de espectrograma e máscaras de limpeza).
* `gerar_espectrograma.py`: Script principal da Etapa 1. Carrega o áudio, aplica a CQT, limpa o ruído com morfologia e plota o comparativo de 4 painéis.

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