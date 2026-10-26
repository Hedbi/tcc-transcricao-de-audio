# =============================================================================
# TCC - Transcrição Automática de Áudio de Guitarra para Tablatura
# Módulo: Análise Exploratória e Visualizações Didáticas
# Arquivo: analise_exploratoria.py
# =============================================================================
"""
Módulo de Análise Exploratória de Áudio de Guitarra para Monografia de TCC.

Este script tem finalidade exclusivamente educacional e exploratória para o
capítulo de fundamentação e análise de características acústicas do TCC.
Ele NÃO faz parte da pipeline central de transcrição em lote, servindo para
gerar figuras de alta qualidade (DPI 300) que ilustram os fenômenos físicos e
algorítmicos essenciais em Recuperação de Informação Musical (MIR):
    1. Representação Tempo-Frequência via Espectrograma CQT (Constant-Q Transform);
    2. Detecção de Transientes de Ataque (Onsets) e correlação temporal com a Forma de Onda;
    3. Rastreamento da Frequência Fundamental (f0) por pYIN sobreposto ao CQT;
    4. Estrutura da Série Harmônica de uma nota de guitarra sustentada (f0, 2f0, 3f0...).

Referências Teóricas:
    - Brown, J. C. (1991). Calculation of a constant Q spectral transform.
      J. Acoust. Soc. Am., 89(1), 425-434.
    - Bello, J. P. et al. (2005). A Tutorial on Onset Detection in Music Signals.
      IEEE Trans. Speech and Audio Processing, 13(5), 1035-1047.
    - Mauch, M., & Dixon, S. (2014). pYIN: A Fundamental Frequency Estimator Using
      Probabilistic YIN. IEEE ICASSP, 659-663.
    - De Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator
      for speech and music. J. Acoust. Soc. Am., 111(4), 1917-1930.
"""

import os
import sys
import argparse
from typing import Tuple, Optional, List

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# =============================================================================
# 1. FUNÇÕES DE PROCESSAMENTO DIGITAL DE SINAIS (DSP / MIR)
# =============================================================================

def carregar_sinal_audio(
    caminho_audio: str,
    sr: int = 22050
) -> Tuple[np.ndarray, int]:
    """
    Carrega o arquivo de áudio mono na taxa de amostragem especificada.

    Parâmetros:
        caminho_audio (str): Caminho para o arquivo de áudio (WAV, MP3, etc.).
        sr (int): Taxa de amostragem de destino (padrão 22.050 Hz).
                  Justificativa: Nyquist = 11.025 Hz, cobrindo com folga a
                  faixa útil musical da guitarra e seus harmônicos relevantes.

    Retorna:
        Tuple[np.ndarray, int]: Vetor 1D do sinal (y) e taxa de amostragem efetiva (sr).
    """
    if not os.path.exists(caminho_audio):
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: '{caminho_audio}'")

    y, taxa = librosa.load(caminho_audio, sr=sr, mono=True)
    return y, taxa


def calcular_espectrograma_cqt(
    y: np.ndarray,
    sr: int = 22050,
    fmin: Optional[float] = None,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    hop_length: int = 512
) -> Tuple[np.ndarray, float]:
    """
    Calcula a Transformada Constant-Q (CQT) e converte as magnitudes para decibéis (dB).

    A CQT mantém a razão Q = f / delta_f constante, gerando faixas de frequência
    logarítmicas geometricamente espaçadas, o que reflete a afinação da escala
    cromática ocidental (12 semitons por oitava).

    Parâmetros:
        y (np.ndarray): Sinal de áudio discreto.
        sr (int): Taxa de amostragem.
        fmin (float, opcional): Frequência mínima central do bin 0.
                                Se None, usa E2 (~82.41 Hz), a nota mais grave
                                da afinação padrão da guitarra de 6 cordas.
        n_bins (int): Quantidade total de bins espectrais (84 bins = 7 oitavas).
        bins_per_octave (int): Resolução cromática (12 semitons por oitava).
        hop_length (int): Avanço temporal em amostras entre frames consecutivos.

    Retorna:
        Tuple[np.ndarray, float]:
            - Matriz 2D de amplitudes em decibéis normalizada (pico = 0 dB).
            - Valor efetivo de fmin em Hz.
    """
    if fmin is None:
        fmin = float(librosa.note_to_hz('E2'))

    # Cálculo da CQT complexa
    cqt_complex = librosa.cqt(
        y=y,
        sr=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length
    )

    # Conversão para escala de magnitude em decibéis relativa ao pico máximo
    cqt_db = librosa.amplitude_to_db(np.abs(cqt_complex), ref=np.max)
    return cqt_db, fmin


def detectar_ataques_onsets(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detecta os instantes temporais de início de notas (onsets) no sinal bruto.

    Utiliza a função de força de ataque (onset strength envelope) baseada no
    fluxo espectral (Spectral Flux) e detecção de picos transientes locais.

    Parâmetros:
        y (np.ndarray): Sinal de áudio discreto.
        sr (int): Taxa de amostragem.
        hop_length (int): Tamanho do salto temporal entre quadros.

    Retorna:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - onset_times: Vetor de tempos (em segundos) de cada ataque detectado.
            - onset_env: Envelope de força de ativação de ataque (1D).
            - env_times: Vetor de tempo correspondente aos quadros do envelope.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    env_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onset_times, onset_env, env_times


def rastrear_f0_pyin(
    y: np.ndarray,
    sr: int = 22050,
    fmin_nota: str = 'C2',
    fmax_nota: str = 'C7',
    hop_length: int = 512,
    limiar_silencio_db: float = -40.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estima a frequência fundamental (f0) do sinal de áudio usando o algoritmo pYIN.

    O pYIN combina a função de diferença média quadrada normalizada (d_t(tau)) do YIN
    com um Modelo Oculto de Markov (HMM) e decodificação de Viterbi, produzindo
    estimativas probabilísticas robustas contra erros de oitava e ruídos.
    Aplica também um filtro de silêncio via energia RMS para descartar ruído de fundo.

    Parâmetros:
        y (np.ndarray): Sinal de áudio discreto.
        sr (int): Taxa de amostragem.
        fmin_nota (str): Nota mínima de busca (padrão 'C2' ≈ 65.4 Hz, abrangendo Drop-D).
        fmax_nota (str): Nota máxima de busca (padrão 'C7' ≈ 2093 Hz, abrangendo até a 24ª casa).
        hop_length (int): Tamanho do salto temporal em amostras.
        limiar_silencio_db (float): Limiar em dB relativo ao pico para rejeitar silêncios/ruído.

    Retorna:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - f0: Vetor de frequências estimadas em Hz para cada quadro.
            - voiced_flag: Vetor booleano indicando presença de vozeamento e energia ativa.
            - times: Vetor de instantes de tempo correspondentes em segundos.
    """
    fmin_hz = float(librosa.note_to_hz(fmin_nota))
    fmax_hz = float(librosa.note_to_hz(fmax_nota))

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y=y,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sr,
        hop_length=hop_length
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # Filtro de energia RMS para descartar falsos positivos em pausas e ruído de fundo
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    mascara_energia = rms_db > limiar_silencio_db

    voiced_flag_filtrado = voiced_flag & mascara_energia
    return f0, voiced_flag_filtrado, times


# =============================================================================
# 2. FUNÇÕES DE VISUALIZAÇÃO GRÁFICA (MATPLOTLIB)
# =============================================================================

def salvar_espectrograma_cqt_simples(
    cqt_db: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    nome_audio: str,
    caminho_saida: str
) -> str:
    """
    Gera e salva o Espectrograma CQT básico com mapa de cores magma e eixos musicais.

    Visualização 1 solicitada no escopo:
    - Espectrograma CQT limpo e didático;
    - Colormap 'magma' com boa diferenciação de intensidade;
    - Eixos claramente identificados: Tempo (s) e Notas Musicais;
    - Título identificando o arquivo de áudio analisado.

    Parâmetros:
        cqt_db (np.ndarray): Matriz CQT em decibéis.
        sr (int): Taxa de amostragem.
        hop_length (int): Passo do janelamento.
        fmin (float): Frequência base da CQT em Hz.
        nome_audio (str): Nome do arquivo de áudio.
        caminho_saida (str): Caminho completo onde a imagem será gravada.

    Retorna:
        str: Caminho do arquivo de imagem salvo.
    """
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)

    img = librosa.display.specshow(
        cqt_db,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        x_axis='time',
        y_axis='cqt_note',
        cmap='magma',
        ax=ax
    )

    ax.set_title(
        f"Espectrograma CQT (Constant-Q Transform) - {nome_audio}",
        fontsize=13,
        fontweight='bold',
        pad=12
    )
    ax.set_xlabel("Tempo (segundos)", fontsize=11, labelpad=8)
    ax.set_ylabel("Nota Musical (Escala Cromática Temperada)", fontsize=11)

    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label("Magnitude Normalizada (dB relativo ao pico)", fontsize=10)

    # Anotação metodológica no rodapé
    fig.text(
        0.12, 0.015,
        "Transformada Constant-Q: 84 bins espectrais (12 bins/oitava, 7 oitavas), fmin ≈ 82.4 Hz (E2).",
        fontsize=8.5,
        color='#444444',
        style='italic'
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(caminho_saida, dpi=300)
    plt.close(fig)
    return caminho_saida


def salvar_comparacao_waveform_onsets(
    y: np.ndarray,
    cqt_db: np.ndarray,
    onset_times: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    nome_audio: str,
    caminho_saida: str
) -> str:
    """
    Gera e salva a comparação entre a Forma de Onda e o Espectrograma CQT com Onsets.

    Visualização 2 solicitada no escopo:
    - Subplot superior: Forma de onda bruta (amplitude vs tempo);
    - Subplot inferior: Espectrograma CQT com marcadores verticais tracejados vermelhos;
    - Demonstração visual do funcionamento da detecção de ataques de palhetada.

    Parâmetros:
        y (np.ndarray): Sinal de áudio no domínio do tempo.
        cqt_db (np.ndarray): Matriz CQT em decibéis.
        onset_times (np.ndarray): Vetor de instantes de tempo dos onsets detectados.
        sr (int): Taxa de amostragem.
        hop_length (int): Passo do janelamento.
        fmin (float): Frequência base da CQT em Hz.
        nome_audio (str): Nome do arquivo de áudio.
        caminho_saida (str): Caminho completo onde a imagem será gravada.

    Retorna:
        str: Caminho do arquivo de imagem salvo.
    """
    fig, (ax_wave, ax_cqt) = plt.subplots(
        2, 1,
        figsize=(12, 7.5),
        dpi=300,
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1.8]}
    )

    # 1. Subplot Superior: Forma de Onda no Domínio do Tempo
    librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='#1f77b4', alpha=0.85)
    ax_wave.set_title(
        f"Alinhamento Temporal: Forma de Onda Bruta vs. Espectrograma CQT - {nome_audio}",
        fontsize=13,
        fontweight='bold',
        pad=10
    )
    ax_wave.set_xlabel("")  # Oculta rótulo intermediário no subplot compartilhado
    ax_wave.set_ylabel("Amplitude Linear", fontsize=10)
    ax_wave.grid(True, linestyle=':', alpha=0.5)

    # Linhas de onset na forma de onda
    for i, t_onset in enumerate(onset_times):
        rotulo = "Onsets Detectados (Palhetadas)" if i == 0 else None
        ax_wave.axvline(
            t_onset,
            color='red',
            linestyle='--',
            linewidth=1.4,
            alpha=0.9,
            label=rotulo
        )
    ax_wave.legend(loc='upper right', framealpha=0.9)

    # 2. Subplot Inferior: Espectrograma CQT com Marcadores de Onset
    img = librosa.display.specshow(
        cqt_db,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        x_axis='time',
        y_axis='cqt_note',
        cmap='magma',
        ax=ax_cqt
    )
    ax_cqt.set_xlabel("Tempo (segundos)", fontsize=11, labelpad=8)
    ax_cqt.set_ylabel("Nota Musical (CQT)", fontsize=10)

    # Linhas de onset no espectrograma
    for t_onset in onset_times:
        ax_cqt.axvline(
            t_onset,
            color='red',
            linestyle='--',
            linewidth=1.4,
            alpha=0.9
        )

    cbar = fig.colorbar(img, ax=ax_cqt, format='%+2.0f dB')
    cbar.set_label("Magnitude (dB)", fontsize=10)

    # Legenda explicativa pedagógica
    texto_info = (
        f"Total de onsets detectados: {len(onset_times)}\n"
        "As linhas verticais tracejadas vermelhas correlacionam o ataque percussivo\n"
        "na forma de onda com o início da emissão espectral no espectrograma CQT."
    )
    ax_cqt.text(
        0.015, 0.06,
        texto_info,
        transform=ax_cqt.transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#cccccc')
    )

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close(fig)
    return caminho_saida


def salvar_sobreposicao_f0_pyin(
    cqt_db: np.ndarray,
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    times: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    nome_audio: str,
    caminho_saida: str
) -> str:
    """
    Gera e salva a sobreposição do rastreamento de F0 (pYIN) sobre o Espectrograma CQT.

    Visualização 3 solicitada no escopo:
    - Espectrograma CQT como fundo;
    - Curva contínua de F0 estimada por pYIN traçada com linha brilhante de alto contraste;
    - Ilustra como o rastreador de pitch acompanha a frequência fundamental sem se perder nos harmônicos.

    Parâmetros:
        cqt_db (np.ndarray): Matriz CQT em decibéis.
        f0 (np.ndarray): Frequências fundamentais estimadas em Hz.
        voiced_flag (np.ndarray): Máscara booleana de quadros com vozeamento.
        times (np.ndarray): Vetor temporal correspondente.
        sr (int): Taxa de amostragem.
        hop_length (int): Passo do janelamento.
        fmin (float): Frequência base da CQT em Hz.
        nome_audio (str): Nome do arquivo de áudio.
        caminho_saida (str): Caminho completo onde a imagem será gravada.

    Retorna:
        str: Caminho do arquivo de imagem salvo.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Espectrograma CQT de fundo
    img = librosa.display.specshow(
        cqt_db,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        x_axis='time',
        y_axis='cqt_note',
        cmap='magma',
        ax=ax
    )

    # Mascarar regiões desvozeadas para evitar linhas horizontais espúrias
    f0_mascarado = np.where(voiced_flag, f0, np.nan)

    # Sobreposição de f0 com linha de alta visibilidade (ciano brilhante)
    # Linha de fundo escura sutil para garantir legibilidade mesmo contra fundos claros
    ax.plot(
        times,
        f0_mascarado,
        color='black',
        linewidth=3.6,
        alpha=0.7,
        label='_nolegend_'
    )
    ax.plot(
        times,
        f0_mascarado,
        color='#00FFFF',
        linewidth=2.2,
        label='Frequência Fundamental Estimada (f0 - pYIN)'
    )

    ax.set_title(
        f"Rastreamento de F0 Monofônico via pYIN sobre Espectrograma CQT - {nome_audio}",
        fontsize=13,
        fontweight='bold',
        pad=12
    )
    ax.set_xlabel("Tempo (segundos)", fontsize=11)
    ax.set_ylabel("Nota Musical (CQT)", fontsize=11)

    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label("Magnitude CQT (dB)", fontsize=10)

    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)

    # Texto pedagógico para a monografia (posicionado no topo onde não há notas fundamentais)
    info_box = (
        "Fundamento Teórico (Mauch & Dixon, 2014):\n"
        "O algoritmo pYIN calcula a função de autocorrelação com diferença normalizada\n"
        "e seleciona múltiplos candidatos a pitch usando uma Cadeia Oculta de Markov (HMM).\n"
        "Isso permite rastrear com precisão a nota raiz (f0), rejeitando os sobretons harmônicos."
    )
    ax.text(
        0.02, 0.96,
        info_box,
        transform=ax.transAxes,
        fontsize=8.5,
        va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#999999')
    )

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close(fig)
    return caminho_saida


def salvar_ilustracao_serie_harmonica(
    y: np.ndarray,
    cqt_db: np.ndarray,
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    times: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    nome_audio: str,
    caminho_saida: str,
    num_harmonics: int = 6
) -> str:
    """
    Gera e salva o diagrama pedagógico da Série Harmônica para uma nota sustentada.

    Visualização 4 solicitada no escopo:
    - Foca em uma nota isolada sustentada no áudio;
    - Zoom temporal no trecho estável de sustentação;
    - Zoom na faixa vertical de frequências dos harmônicos;
    - Anotações com setas e caixas indicando f0, 2f0, 3f0, 4f0, 5f0, 6f0;
    - Diagrama educacional indispensável para fundamentar os desafios de MIR no TCC.

    Parâmetros:
        y (np.ndarray): Sinal de áudio discreto.
        cqt_db (np.ndarray): Matriz CQT em decibéis.
        f0 (np.ndarray): Frequências fundamentais estimadas em Hz.
        voiced_flag (np.ndarray): Máscara de vozeamento.
        times (np.ndarray): Vetor temporal em segundos.
        sr (int): Taxa de amostragem.
        hop_length (int): Passo do janelamento.
        fmin (float): Frequência base da CQT em Hz.
        nome_audio (str): Nome do arquivo de áudio.
        caminho_saida (str): Caminho completo onde a imagem será gravada.
        num_harmonics (int): Número de harmônicos a ilustrar (padrão: 6).

    Retorna:
        str: Caminho do arquivo de imagem salvo.
    """
    # 1. Localização do segmento sustentado mais longo e estável
    indices_vozeados = np.where(voiced_flag)[0]

    if len(indices_vozeados) == 0:
        # Fallback caso o áudio não possua trecho vozeado nítido
        t_inicio = 0.5
        t_fim = min(2.5, len(y) / sr)
        f0_referencia = float(librosa.note_to_hz('E2'))
    else:
        # Agrupar índices contínuos de vozeamento
        diferencas = np.diff(indices_vozeados)
        cortes = np.where(diferencas > 1)[0]
        segmentos = np.split(indices_vozeados, cortes + 1)
        segmento_mais_longo = max(segmentos, key=len)

        t_inicio = times[segmento_mais_longo[0]]
        t_fim = times[segmento_mais_longo[-1]]
        duracao = t_fim - t_inicio

        f0_valores = f0[segmento_mais_longo]
        f0_referencia = float(np.median(f0_valores))

        # Se o segmento for longo, isolar uma janela central estável de ~2.0 segundos
        if duracao > 2.2:
            t_inicio = t_inicio + 0.3
            t_fim = t_inicio + 2.0
        elif duracao < 0.6:
            # Garante uma janela mínima visível
            margem = (0.8 - duracao) / 2
            t_inicio = max(0.0, t_inicio - margem)
            t_fim = min(float(len(y) / sr), t_fim + margem)

    nota_fundamental = librosa.hz_to_note(f0_referencia, unicode=False)

    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=300)

    img = librosa.display.specshow(
        cqt_db,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        x_axis='time',
        y_axis='cqt_note',
        cmap='magma',
        ax=ax
    )

    # 2. Definição do zoom temporal e espectral
    ax.set_xlim(t_inicio, t_fim)

    # Limites verticais focados nos primeiros N harmônicos
    f_min_zoom = max(70.0, f0_referencia * 0.85)
    f_max_zoom = min(sr / 2.0, (num_harmonics + 0.8) * f0_referencia)
    ax.set_ylim(f_min_zoom, f_max_zoom)

    # 3. Anotação didática dos harmônicos com setas
    descricoes_intervalos = [
        "Fundamental (f0)",
        "2º Harmônico (1ª Oitava, 2f0)",
        "3º Harmônico (5ª Justa, 3f0)",
        "4º Harmônico (2ª Oitava, 4f0)",
        "5º Harmônico (3ª Maior, 5f0)",
        "6º Harmônico (5ª Justa, 6f0)",
        "7º Harmônico (7ª Menor, 7f0)",
        "8º Harmônico (3ª Oitava, 8f0)"
    ]

    dur_zoom = t_fim - t_inicio

    for k in range(1, num_harmonics + 1):
        fk = k * f0_referencia
        nota_k = librosa.hz_to_note(fk, unicode=False)

        desc = descricoes_intervalos[k - 1] if k - 1 < len(descricoes_intervalos) else f"{k}º Harmônico ({k}f0)"
        texto_anotacao = f"{k}f0 = {fk:.1f} Hz ({nota_k})\n{desc}"

        # Alternar o posicionamento horizontal para evitar sobreposição em frequências próximas
        if k % 2 == 1:
            t_alvo = t_inicio + 0.20 * dur_zoom
            t_texto = t_inicio + 0.48 * dur_zoom
        else:
            t_alvo = t_inicio + 0.52 * dur_zoom
            t_texto = t_inicio + 0.78 * dur_zoom

        # Apenas desenha se estiver dentro dos limites de frequência da visualização
        if f_min_zoom <= fk <= f_max_zoom:
            ax.annotate(
                texto_anotacao,
                xy=(t_alvo, fk),
                xytext=(t_texto, fk),
                arrowprops=dict(
                    facecolor='#00FFCC',
                    edgecolor='#111111',
                    arrowstyle='->',
                    lw=2.0
                ),
                fontsize=8.5,
                fontweight='bold',
                color='#111111',
                va='center',
                ha='left',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='#FFFFF0',
                    alpha=0.92,
                    edgecolor='#555555'
                )
            )

    ax.set_title(
        f"Ilustração da Série Harmônica em Nota Sustentada ({nota_fundamental}) - {nome_audio}",
        fontsize=13,
        fontweight='bold',
        pad=12
    )
    ax.set_xlabel("Tempo (segundos)", fontsize=11, labelpad=8)
    ax.set_ylabel("Nota Musical (CQT)", fontsize=11)

    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label("Magnitude (dB)", fontsize=10)

    # Caixa explicativa da física acústica e desafio de MIR
    box_acustica = (
        "Fenômeno Físico da Série Harmônica:\n"
        "A vibração transversal de uma corda gera modos ressonantes naturais em múltiplos inteiros (k * f0).\n"
        "Desafio para MIR: Algoritmos simplistas de detecção de picos espectrais interpretam esses harmônicos\n"
        "paralelos como notas adicionais tocadas simultaneamente (falsos acordes). Rastreamentos no domínio\n"
        "do tempo (pYIN) ou peneiras harmônicas (Harmonic Sieve) são indispensáveis para isolar a nota raiz."
    )
    fig.text(
        0.10, 0.015,
        box_acustica,
        fontsize=8.5,
        color='#222222',
        bbox=dict(boxstyle='square,pad=0.4', facecolor='#F4F4F4', edgecolor='#B0B0B0')
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(caminho_saida, dpi=300)
    plt.close(fig)
    return caminho_saida


# =============================================================================
# 3. PIPELINE DE EXECUÇÃO COMPLETA
# =============================================================================

def executar_analise_exploratoria(
    caminho_audio: str,
    pasta_saida: str = "resultados/exploratoria",
    sr: int = 22050,
    hop_length: int = 512,
    fmin_nota: str = "E2",
    n_bins: int = 84
) -> List[str]:
    """
    Executa a análise exploratória completa de um arquivo de áudio e salva as 4 figuras.

    Parâmetros:
        caminho_audio (str): Caminho do arquivo de áudio de entrada.
        pasta_saida (str): Diretório de destino para gravação das imagens.
        sr (int): Taxa de amostragem (padrão: 22050 Hz).
        hop_length (int): Passo do janelamento (padrão: 512 amostras).
        fmin_nota (str): Nota mínima da CQT (padrão: 'E2').
        n_bins (int): Quantidade total de bins da CQT (padrão: 84).

    Retorna:
        List[str]: Lista com os caminhos absolutos das 4 imagens geradas.
    """
    if not os.path.exists(caminho_audio):
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: '{caminho_audio}'")

    os.makedirs(pasta_saida, exist_ok=True)
    nome_arquivo = os.path.basename(caminho_audio)
    nome_base, _ = os.path.splitext(nome_arquivo)

    print(f"\n=================================================================")
    print(f" INICIANDO ANÁLISE EXPLORATÓRIA: {nome_arquivo}")
    print(f"=================================================================")
    print(f"[*] Parâmetros:")
    print(f"    - Taxa de amostragem (sr): {sr} Hz")
    print(f"    - Hop length: {hop_length} amostras (~{hop_length/sr*1000:.1f} ms/frame)")
    print(f"    - CQT fmin: {fmin_nota} ({librosa.note_to_hz(fmin_nota):.2f} Hz)")
    print(f"    - Bins CQT: {n_bins} (7 oitavas)")
    print(f"    - Diretório de saída: {pasta_saida}")

    # 1. Carregamento do Sinal de Áudio
    print(f"\n[1/5] Carregando sinal de áudio...")
    y, sr_efetivo = carregar_sinal_audio(caminho_audio, sr=sr)
    duracao_seg = len(y) / sr_efetivo
    print(f"    -> Duração: {duracao_seg:.2f} segundos ({len(y)} amostras)")

    # 2. Cálculo da Transformada Constant-Q (CQT)
    print(f"[2/5] Calculando Transformada Constant-Q (CQT em dB)...")
    fmin_hz = float(librosa.note_to_hz(fmin_nota))
    cqt_db, fmin_efetivo = calcular_espectrograma_cqt(
        y=y,
        sr=sr_efetivo,
        fmin=fmin_hz,
        n_bins=n_bins,
        hop_length=hop_length
    )
    print(f"    -> Dimensões da matriz CQT: {cqt_db.shape[0]} bins x {cqt_db.shape[1]} frames")

    # 3. Detecção de Onsets (Ataques)
    print(f"[3/5] Detectando transientes de ataque (Onsets)...")
    onset_times, onset_env, env_times = detectar_ataques_onsets(
        y=y,
        sr=sr_efetivo,
        hop_length=hop_length
    )
    print(f"    -> Ataques identificados: {len(onset_times)} eventos temporais")
    if len(onset_times) > 0:
        tempos_str = ", ".join([f"{t:.2f}s" for t in onset_times[:6]])
        if len(onset_times) > 6:
            tempos_str += f" ... (+{len(onset_times)-6} outros)"
        print(f"    -> Instantes: [{tempos_str}]")

    # 4. Rastreamento de Frequência Fundamental (pYIN)
    print(f"[4/5] Rastreando frequência fundamental (f0) via pYIN...")
    f0, voiced_flag, times_pyin = rastrear_f0_pyin(
        y=y,
        sr=sr_efetivo,
        fmin_nota='C2',
        fmax_nota='C7',
        hop_length=hop_length
    )
    qtd_vozeados = int(np.sum(voiced_flag))
    print(f"    -> Quadros vozeados detectados: {qtd_vozeados} de {len(f0)} (~{qtd_vozeados/len(f0)*100:.1f}%)")
    if qtd_vozeados > 0:
        f0_mediana = float(np.median(f0[voiced_flag]))
        nota_mediana = librosa.hz_to_note(f0_mediana, unicode=False)
        print(f"    -> Mediana de f0 em trechos ativos: {f0_mediana:.1f} Hz (Nota: {nota_mediana})")

    # 5. Geração e Salvamento das 4 Visualizações
    print(f"\n[5/5] Renderizando e salvando gráficos didáticos em '{pasta_saida}'...")
    arquivos_gerados = []

    # Visualização 1: Espectrograma CQT Limpo
    out_fig1 = os.path.join(pasta_saida, f"{nome_base}_1_espectrograma_cqt.png")
    salvar_espectrograma_cqt_simples(
        cqt_db=cqt_db,
        sr=sr_efetivo,
        hop_length=hop_length,
        fmin=fmin_efetivo,
        nome_audio=nome_arquivo,
        caminho_saida=out_fig1
    )
    arquivos_gerados.append(out_fig1)
    print(f"    [OK] Imagem 1 salva: {out_fig1}")

    # Visualização 2: Forma de Onda vs Onsets
    out_fig2 = os.path.join(pasta_saida, f"{nome_base}_2_waveform_onsets.png")
    salvar_comparacao_waveform_onsets(
        y=y,
        cqt_db=cqt_db,
        onset_times=onset_times,
        sr=sr_efetivo,
        hop_length=hop_length,
        fmin=fmin_efetivo,
        nome_audio=nome_arquivo,
        caminho_saida=out_fig2
    )
    arquivos_gerados.append(out_fig2)
    print(f"    [OK] Imagem 2 salva: {out_fig2}")

    # Visualização 3: Sobreposição pYIN F0 Tracking
    out_fig3 = os.path.join(pasta_saida, f"{nome_base}_3_pyin_f0_tracking.png")
    salvar_sobreposicao_f0_pyin(
        cqt_db=cqt_db,
        f0=f0,
        voiced_flag=voiced_flag,
        times=times_pyin,
        sr=sr_efetivo,
        hop_length=hop_length,
        fmin=fmin_efetivo,
        nome_audio=nome_arquivo,
        caminho_saida=out_fig3
    )
    arquivos_gerados.append(out_fig3)
    print(f"    [OK] Imagem 3 salva: {out_fig3}")

    # Visualização 4: Ilustração da Série Harmônica
    out_fig4 = os.path.join(pasta_saida, f"{nome_base}_4_serie_harmonica.png")
    salvar_ilustracao_serie_harmonica(
        y=y,
        cqt_db=cqt_db,
        f0=f0,
        voiced_flag=voiced_flag,
        times=times_pyin,
        sr=sr_efetivo,
        hop_length=hop_length,
        fmin=fmin_efetivo,
        nome_audio=nome_arquivo,
        caminho_saida=out_fig4,
        num_harmonics=6
    )
    arquivos_gerados.append(out_fig4)
    print(f"    [OK] Imagem 4 salva: {out_fig4}")

    print(f"\n[SUCESSO] Todas as 4 visualizações foram geradas com sucesso!")
    return arquivos_gerados


# =============================================================================
# 4. INTERFACE DE LINHA DE COMANDO (CLI)
# =============================================================================

def criar_analisador_argumentos() -> argparse.ArgumentParser:
    """
    Constrói o analisador de argumentos de linha de comando com descrições em português.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Análise Exploratória e Visualizações Didáticas para Monografia de TCC em MIR.\n"
            "Gera figuras em alta resolução (300 DPI) para o capítulo de fundamentação acústica,\n"
            "incluindo espectrograma CQT, detecção de onsets, curva pYIN e série harmônica."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "audio",
        nargs="?",
        default="audios/som4.mp3",
        help="Caminho do arquivo de áudio de entrada (ex.: audios/som4.mp3 ou audios/som2.wav)."
    )
    parser.add_argument(
        "-a", "--audio",
        dest="audio_flag",
        default=None,
        help="Caminho alternativo para o arquivo de áudio via flag."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="resultados/exploratoria",
        help="Diretório onde as imagens geradas serão salvas."
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Taxa de amostragem (Hz) para carregamento do sinal de áudio."
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Tamanho do avanço de janela (amostras) entre quadros de análise."
    )
    parser.add_argument(
        "--fmin",
        default="E2",
        help="Nota musical de referência para a frequência mínima da CQT."
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=84,
        help="Quantidade total de bins na escala logarítmica da CQT (84 bins = 7 oitavas)."
    )

    return parser


def main():
    """
    Função principal de entrada via linha de comando.
    """
    parser = criar_analisador_argumentos()
    args = parser.parse_args()

    caminho_audio = args.audio_flag if args.audio_flag else args.audio

    try:
        executar_analise_exploratoria(
            caminho_audio=caminho_audio,
            pasta_saida=args.output_dir,
            sr=args.sr,
            hop_length=args.hop_length,
            fmin_nota=args.fmin,
            n_bins=args.n_bins
        )
    except Exception as erro:
        print(f"\n[ERRO FATAL] Falha durante a análise exploratória: {erro}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
