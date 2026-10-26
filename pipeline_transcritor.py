# =============================================================================
# TCC - Transcrição Automática de Áudio de Guitarra para Tablatura
# Módulo Central: Pipeline de Transcrição Monofônica de Notas
# =============================================================================
"""
Pipeline de Transcrição de Áudio de Guitarra (librosa.onset_detect + pYIN)
--------------------------------------------------------------------------
Este módulo implementa a nova arquitetura central para transcrição monofônica
de gravações de guitarra, operando diretamente sobre o sinal temporal bruto
(waveform) e substituindo as antigas operações morfológicas sobre máscaras CQT.

ARQUITETURA DO PIPELINE:
  audio.wav (.mp3)
      │
      ▼
  1. Carregamento do Sinal de Áudio Bruto (sr = 22050 Hz)
      │
      ▼
  2. Detecção de Onsets no Sinal Temporal (librosa.onset.onset_detect)
      │
      ▼
  3. Segmentação Temporal em Notas [onset_i, offset_i]
      │
      ▼
  4. Rastreamento de Pitch (pYIN) por Segmento
      │
      ▼
  5. Agregação Estatística (Mediana f0) & Conversão (Hz → MIDI → Nota Musical)
      │
      ▼
  6. Emissão de Tabela Formatada & Lista Estruturada de Dicionários
      │
      ▼ (Opcional via flag --visualizar)
  7. Visualização Espectral CQT com Marcadores Temporais de Onset

JUSTIFICATIVA ACADÊMICA DA MUDANÇA DE PARADIGMA:
-------------------------------------------------
Na versão inicial do projeto, o espectrograma CQT era submetido a limiarização
rígida (-40 dB) seguida de abertura e fechamento morfológico bidimensional.
Conforme apontado na avaliação docente, essa abordagem apresentava falhas críticas:
  a) A abertura morfológica com elemento estruturante horizontal erodia
     precisamente os transientes verticais de alta energia (ataques de palheta),
     gerando perda de precisão temporal e atraso artificial nos instantes de início.
  b) Notas consecutivas de mesma altura ou com ataques rápidos eram fundidas
     pelo fechamento morfológico, inviabilizando a separação confiável de eventos.
  c) A binarização descartava informações contínuas essenciais para o rastreamento
     preciso da frequência fundamental (f0).

A solução adotada neste pipeline alinha-se aos padrões de Recuperação de Informação
Musical (MIR - Music Information Retrieval):
  1. Detecção temporal no sinal contínuo via Spectral Flux / Onset Strength Envelope,
     preservando a acurácia dos ataques transientes.
  2. Rastreamento de f0 via pYIN (Probabilistic YIN - Mauch & Dixon, 2014), que
     combina a função de diferença média quadrada normalizada com Modelos Ocultos
     de Markov (HMM) e decodificação de Viterbi, suprimindo erros de salto de oitava.
  3. Manutenção da CQT exclusivamente como ferramenta de inspeção visual exploratória.

Autor: Desenvolvido para o Trabalho de Conclusão de Curso (TCC).
"""

import os
import sys
import glob
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# =============================================================================
# ETAPA 1: CARREGAMENTO E PRÉ-PROCESSAMENTO DO ÁUDIO
# =============================================================================
def carregar_audio(caminho_audio: str, sr: int = 22050) -> tuple[np.ndarray, int, float]:
    """
    Carrega o arquivo de áudio mono e calcula sua duração temporal total.

    Parâmetros:
        caminho_audio (str): Caminho para o arquivo de áudio (.wav, .mp3, etc.).
        sr (int): Taxa de amostragem alvo em Hertz. Padrão: 22050 Hz.

    Retorna:
        tuple[np.ndarray, int, float]:
            - y: Sinal temporal de áudio normalizado em ponto flutuante [-1.0, 1.0].
            - sr: Taxa de amostragem efetiva utilizada.
            - duracao_s: Duração total do áudio em segundos.

    Justificativa Teórica (Taxa de Amostragem sr=22050 Hz):
        O Teorema de Nyquist-Shannon estabelece que a taxa de amostragem deve
        ser de pelo menos o dobro da componente espectral máxima de interesse.
        Na afinação padrão da guitarra (E2 ~ 82,41 Hz a E6 ~ 1318,51 Hz no 24º
        traste), mesmo considerando os harmônicos superiores acústicos mais
        relevantes do timbre da corda (até ~8.000 Hz), uma amostragem de 22.050 Hz
        fornece uma frequência de corte de Nyquist em 11.025 Hz. Isso cobre com
        folga toda a informação tonal necessária, ao mesmo tempo em que reduz o
        volume de dados pela metade frente aos 44.100 Hz de estúdio, acelerando
        significativamente o processamento dos algoritmos subsequentes.
    """
    if not os.path.exists(caminho_audio):
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {caminho_audio}")

    # librosa.load converte automaticamente para mono e normaliza em [-1.0, 1.0]
    y, sr = librosa.load(caminho_audio, sr=sr, mono=True)
    duracao_s = float(len(y) / sr)

    return y, sr, duracao_s


# =============================================================================
# ETAPA 2: DETECÇÃO DE ONSETS E CÁLCULO DE OFFSETS
# =============================================================================
def detectar_onsets_offsets(y: np.ndarray, sr: int = 22050) -> tuple[np.ndarray, np.ndarray]:
    """
    Detecta os instantes temporais de início de cada nota (onsets) e calcula
    os instantes correspondentes de término (offsets).

    Parâmetros:
        y (np.ndarray): Sinal de áudio unidimensional.
        sr (int): Taxa de amostragem em Hz.

    Retorna:
        tuple[np.ndarray, np.ndarray]:
            - onsets (np.ndarray): Vetor de instantes de onset em segundos.
            - offsets (np.ndarray): Vetor de instantes de offset em segundos.

    Justificativa Teórica (Detecção de Onset no Sinal Bruto vs. Máscara CQT):
        A função `librosa.onset.onset_detect` calcula o envelope de força de onset
        (Onset Strength Envelope) a partir do fluxo espectral positivo (Spectral Flux).
        Transientes de palhetada geram um surto instantâneo de energia em múltiplas
        faixas de frequência simultâneas. Detectar os picos desse envelope diretamente
        no sinal temporal preserva a integridade física dos ataques com precisão de
        milissegundos, evitando a distorção introduzida por filtros morfológicos de
        imagem que dilatam ou corroem regiões de contorno.

        Regra de Offsets:
        Em transcrição monofônica com notas encadeadas ou sustentadas, o término
        (offset) da nota 'i' é definido como o início (onset) da nota subsequente
        'i+1'. Para a última nota do sinal, o offset corresponde à duração total do áudio.
    """
    # Detecção de instantes de ataque diretamente em segundos
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')

    duracao_total = len(y) / sr

    # Tratamento caso nenhum onset seja identificado (ex: áudio estático ou silencioso)
    if len(onsets) == 0:
        return np.array([]), np.array([])

    # Cálculo dos offsets correspondentes
    # Cada nota encerra no onset seguinte; a última encerra na duração final do áudio
    offsets = np.empty_like(onsets)
    if len(onsets) > 1:
        offsets[:-1] = onsets[1:]
    offsets[-1] = duracao_total

    return onsets, offsets


# =============================================================================
# ETAPA 3: RASTREAMENTO DE PITCH (F0) POR SEGMENTO VIA pYIN
# =============================================================================
def estimar_pitch_segmento(
    y_segmento: np.ndarray,
    sr: int = 22050,
    fmin: float = None,
    fmax: float = None
) -> tuple[float, int | None, str]:
    """
    Estima a frequência fundamental (f0) dominante de um segmento sonoro delimitado
    entre um onset e seu respectivo offset, convertendo o resultado para escala MIDI
    e nome de nota cromática.

    Parâmetros:
        y_segmento (np.ndarray): Recorte temporal do sinal correspondente à nota.
        sr (int): Taxa de amostragem do áudio.
        fmin (float, opcional): Limite inferior de busca de frequência fundamental (Hz).
                                Padrão: ~82,41 Hz (E2 - 6ª corda solta).
        fmax (float, opcional): Limite superior de busca de frequência fundamental (Hz).
                                Padrão: ~1318,51 Hz (E6 - casa 24 da 1ª corda).

    Retorna:
        tuple[float, int | None, str]:
            - freq_mediana (float): Frequência fundamental representativa (Hz) ou np.nan.
            - midi_numero (int | None): Número da nota na escala MIDI (0-127) ou None.
            - nome_nota (str): Notação científica da nota (ex: 'E2', 'A4') ou 'N/A'.

    Justificativa Teórica (Algoritmo pYIN e Filtragem por Mediana):
        1. O algoritmo pYIN (Mauch & Dixon, 2014) estende o consagrado algoritmo YIN
           (de Cheveigné & Kawahara, 2002) ao tratar múltiplos limiares de aperiodismo
           através de uma distribuição probabilística observada por um Modelo Oculto
           de Markov (HMM). Essa abordagem reduz drasticamente erros de dobra ou
           subdivisão de oitava (Octave Errors), comuns em instrumentos de corda com
           ressonância harmônica densa.
        2. fmin = note_to_hz('E2') descarta zumbidos elétricos (como 60 Hz da rede AC)
           e batidas mecânicas abaixo da extensão tocável da afinação padrão da guitarra.
        3. A mediana dos valores f0 não-NaN é aplicada como métrica de tendência
           central robusta: diferentemente da média aritmética, a mediana é imune
           ao "chirp" transitório da palhetada no início do segmento e a oscilações
           periódicas de vibrato durante a sustentação da nota.
        4. Conversão Hz → MIDI:
           fórmula: MIDI = 69 + 12 * log2(f0 / 440.0)
           baseada no padrão internacional ISO 16 (Afinação de concerto A4 = 440 Hz).
    """
    if fmin is None:
        fmin = float(librosa.note_to_hz('E2'))  # ~82.41 Hz
    if fmax is None:
        fmax = float(librosa.note_to_hz('E6'))  # ~1318.51 Hz

    # Caso o segmento seja vazio ou excessivamente curto
    if len(y_segmento) == 0:
        return np.nan, None, "N/A"

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_segmento,
            fmin=fmin,
            fmax=fmax,
            sr=sr
        )
    except Exception as erro:
        # Prevenção contra eventuais anomalias numéricas em segmentos espúrios
        return np.nan, None, "N/A"

    # Seleção estrita de frames sonoros (não-NaN e classificados como voiced)
    f0_validos = f0[~np.isnan(f0)]

    if len(f0_validos) == 0:
        return np.nan, None, "N/A"

    # Mediana robusta da frequência fundamental no segmento sustentado
    freq_mediana = float(np.median(f0_validos))

    if freq_mediana <= 0.0 or np.isnan(freq_mediana):
        return np.nan, None, "N/A"

    # Conversão rigorosa Hz -> MIDI semitons contínuos
    midi_continuo = 69.0 + 12.0 * np.log2(freq_mediana / 440.0)
    midi_numero = int(round(midi_continuo))

    # Conversão MIDI -> Notação Cromática Científica (C4, A2, etc.)
    nome_nota = librosa.midi_to_note(midi_numero, unicode=False)

    return freq_mediana, midi_numero, nome_nota


def estimar_f0_audio(
    y: np.ndarray,
    sr: int = 22050,
    fmin: float | None = None,
    fmax: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estima f0 uma única vez sobre o sinal original.

    Rodar pYIN separadamente em cada trecho entre onsets descarta o contexto
    necessário para a janela de análise e falha principalmente em notas curtas.
    O rastreamento global preserva esse contexto; a agregação por nota é feita
    posteriormente usando os intervalos de onset.
    """
    if fmin is None:
        fmin = float(librosa.note_to_hz('E2'))
    if fmax is None:
        fmax = float(librosa.note_to_hz('E6'))

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr
    )
    return f0, voiced_flag


def resumir_f0_segmento(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    sr: int,
    t_onset: float,
    t_offset: float
) -> tuple[float, int | None, str]:
    """Resume o f0 global no intervalo temporal de uma nota."""
    tempos = librosa.times_like(f0, sr=sr)
    dentro_do_intervalo = (tempos >= t_onset) & (tempos < t_offset)
    validos = f0[dentro_do_intervalo & voiced_flag & ~np.isnan(f0)]

    if len(validos) == 0:
        return np.nan, None, "N/A"

    freq_mediana = float(np.median(validos))
    midi_continuo = float(librosa.hz_to_midi(freq_mediana))
    midi_numero = int(round(midi_continuo))
    nome_nota = librosa.midi_to_note(midi_numero, unicode=False)
    return freq_mediana, midi_numero, nome_nota


# =============================================================================
# ETAPA 4: ORQUESTRAÇÃO DO PIPELINE DE TRANSCRIÇÃO
# =============================================================================
def transcrever_audio(caminho_audio: str, sr: int = 22050) -> list[dict]:
    """
    Executa o fluxo completo de transcrição monofônica: carregamento, detecção
    de onsets, segmentação temporal, estimativa de f0 via pYIN e formatação dos
    dados musicais detectados.

    Parâmetros:
        caminho_audio (str): Caminho para o arquivo de áudio a ser transcrito.
        sr (int): Taxa de amostragem padrão (22050 Hz).

    Retorna:
        list[dict]: Lista de dicionários, contendo para cada nota detectada:
            - 'nota_idx' (int): Número sequencial da nota (1-indexado).
            - 'onset_time' (float): Instante inicial em segundos.
            - 'offset_time' (float): Instante final em segundos.
            - 'duration' (float): Duração total do evento em segundos.
            - 'freq_hz' (float | None): Frequência fundamental em Hz.
            - 'midi_number' (int | None): Número inteiro da nota MIDI.
            - 'note_name' (str): Nome cromático da nota (ex: 'E2', 'A2').
            - Aliases auxiliares em português ('nota', 'onset', 'offset', 'duracao').
    """
    print(f"\n[1/3] Carregando áudio: {os.path.basename(caminho_audio)}")
    y, sr, duracao_total = carregar_audio(caminho_audio, sr=sr)
    print(f"      Duração total: {duracao_total:.3f} s | Taxa de amostragem: {sr} Hz")

    print("[2/3] Executando detecção de onsets (librosa.onset.onset_detect)...")
    onsets, offsets = detectar_onsets_offsets(y, sr)
    total_onsets = len(onsets)

    print(f"      Onsets identificados: {total_onsets}")
    if total_onsets > 0:
        onsets_formatados = np.array2string(onsets, precision=3, separator=', ', suppress_small=True)
        print(f"      Instantes temporais (s):\n      {onsets_formatados}")
    else:
        print("      [AVISO] Nenhum evento de ataque transitório foi identificado no sinal.")
        return []

    print("[3/3] Rastreando f0 no sinal original com pYIN...")
    f0_global, voiced_global = estimar_f0_audio(y, sr=sr)
    lista_notas = []

    for i in range(total_onsets):
        t_onset = float(onsets[i])
        t_offset = float(offsets[i])
        duracao = t_offset - t_onset

        freq_hz, midi_num, nome_nota = resumir_f0_segmento(
            f0_global, voiced_global, sr, t_onset, t_offset
        )

        registro_nota = {
            'nota_idx': i + 1,
            'onset_time': round(t_onset, 3),
            'offset_time': round(t_offset, 3),
            'duration': round(duracao, 3),
            'freq_hz': round(freq_hz, 2) if not np.isnan(freq_hz) else None,
            'midi_number': midi_num,
            'note_name': nome_nota,
            # Aliases em português para integração flexível
            'nota': nome_nota,
            'onset': round(t_onset, 3),
            'offset': round(t_offset, 3),
            'duracao': round(duracao, 3),
        }
        lista_notas.append(registro_nota)

    return lista_notas


# =============================================================================
# ETAPA 5: APRESENTAÇÃO E FORMATAÇÃO DA SAÍDA EM TABELA
# =============================================================================
def imprimir_tabela_notas(lista_notas: list[dict], nome_arquivo: str = ""):
    """
    Exibe no console uma tabela formatada com todas as notas detectadas,
    suas frequências fundamentais, números MIDI, instantes de onset/offset e durações.

    Parâmetros:
        lista_notas (list[dict]): Lista de notas gerada pelo pipeline.
        nome_arquivo (str): Nome do arquivo analisado para compor o título.
    """
    separador_largo = "=" * 88
    separador_fino = "-" * 88

    print("\n" + separador_largo)
    titulo = f"RESULTADO DA TRANSCRIÇÃO DE ÁUDIO"
    if nome_arquivo:
        titulo += f": {nome_arquivo}"
    print(f"{titulo:^88}")
    print(separador_largo)

    cabecalho = (
        f"{'Note#':^7} | {'Note Name':^11} | {'Freq (Hz)':^11} | "
        f"{'MIDI':^8} | {'Onset (s)':^11} | {'Offset (s)':^11} | {'Duration (s)':^12}"
    )
    print(cabecalho)
    print(separador_fino)

    if not lista_notas:
        print(f"{'Nenhuma nota identificada para exibição.':^88}")
        print(separador_largo + "\n")
        return

    for item in lista_notas:
        idx = item['nota_idx']
        nome = item['note_name']
        freq = f"{item['freq_hz']:.2f}" if item['freq_hz'] is not None else "---"
        midi = f"{item['midi_number']:3d}" if item['midi_number'] is not None else "---"
        onset = f"{item['onset_time']:.3f}"
        offset = f"{item['offset_time']:.3f}"
        dur = f"{item['duration']:.3f}"

        linha = (
            f"{idx:^7d} | {nome:^11s} | {freq:^11s} | "
            f"{midi:^8s} | {onset:^11s} | {offset:^11s} | {dur:^12s}"
        )
        print(linha)

    print(separador_largo)
    print(f"Total de eventos musicais transcritos: {len(lista_notas)}")
    print(separador_largo + "\n")


# =============================================================================
# ETAPA 6: VISUALIZAÇÃO EXPLORATÓRIA DO ESPECTROGRAMA CQT (OPCIONAL)
# =============================================================================
def obter_proximo_caminho_resultado(pasta_resultados: str) -> str:
    """
    Verifica os arquivos já existentes no diretório de saída e gera o próximo
    nome de arquivo sequencial no padrão 'resultado_N.png'.

    Parâmetros:
        pasta_resultados (str): Caminho do diretório de resultados.

    Retorna:
        str: Caminho absoluto completo para o novo arquivo de imagem.
    """
    os.makedirs(pasta_resultados, exist_ok=True)
    arquivos_existentes = glob.glob(os.path.join(pasta_resultados, 'resultado_*.png'))

    numeros = []
    for arq in arquivos_existentes:
        try:
            nome_base = os.path.basename(arq)
            num = int(nome_base.replace('resultado_', '').replace('.png', ''))
            numeros.append(num)
        except ValueError:
            pass

    proximo_numero = 1 if not numeros else max(numeros) + 1
    return os.path.join(pasta_resultados, f'resultado_{proximo_numero}.png')


def gerar_visualizacao_cqt(
    caminho_audio: str,
    caminho_saida: str = None,
    pasta_resultados: str = None,
    sr: int = 22050
) -> str:
    """
    Gera e salva o espectrograma CQT do sinal sonoro com linhas verticais
    demarcando com precisão os instantes de onset detectados.

    Parâmetros:
        caminho_audio (str): Caminho do áudio de entrada.
        caminho_saida (str, opcional): Caminho específico onde a imagem será salva.
        pasta_resultados (str, opcional): Diretório onde gerar arquivo sequencial.
        sr (int): Taxa de amostragem em Hz.

    Retorna:
        str: Caminho completo onde a imagem gerada foi armazenada.

    Justificativa Teórica (Papel da CQT como Ferramenta Visual Exploratória):
        A Transformada de Q Constante (CQT - Brown, 1991) projeta o sinal em uma escala
        espectral geometricamente espaçada (12 bins por oitava), alinhando cada bin
        a um semitom da escala temperada ocidental. Sobrepor os onsets temporalmente
        validados permite ao pesquisador conferir visualmente a coerência entre o
        ataque físico (início da mancha de energia nos harmônicos) e a resposta do
        algoritmo de novidade espectral, cumprindo função de diagnóstico analítico.
    """
    y, sr, _ = carregar_audio(caminho_audio, sr=sr)
    onsets, _ = detectar_onsets_offsets(y, sr)

    fmin = float(librosa.note_to_hz('E2'))  # Mi mais grave da guitarra (~82.41 Hz)
    cqt = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=84, hop_length=512)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    fig, ax = plt.subplots(figsize=(14, 6))

    img = librosa.display.specshow(
        cqt_db,
        sr=sr,
        x_axis='time',
        y_axis='cqt_note',
        fmin=fmin,
        hop_length=512,
        ax=ax,
        cmap='magma'
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Intensidade Espectral (dB)')

    # Sobreposição dos onsets com linhas verticais vermelhas tracejadas
    for idx, t_onset in enumerate(onsets):
        legenda = 'Onsets Detectados' if idx == 0 else ""
        ax.axvline(
            x=t_onset,
            color='red',
            linestyle='--',
            linewidth=1.2,
            alpha=0.85,
            label=legenda
        )

    nome_arquivo = os.path.basename(caminho_audio)
    ax.set_title(
        f'Espectrograma CQT com Marcadores Temporais de Onset: {nome_arquivo}',
        fontsize=13,
        fontweight='bold',
        pad=12
    )
    ax.set_xlabel('Tempo (segundos)', fontsize=11)
    ax.set_ylabel('Nota Musical (CQT)', fontsize=11)

    if len(onsets) > 0:
        ax.legend(loc='upper right', framealpha=0.85)

    plt.tight_layout()

    # Determinação do caminho de saída
    if caminho_saida is None:
        if pasta_resultados is None:
            pasta_resultados = os.path.join(os.path.dirname(os.path.abspath(caminho_audio)), '..', 'resultados')
        caminho_saida = obter_proximo_caminho_resultado(os.path.abspath(pasta_resultados))

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)

    return caminho_saida


# =============================================================================
# RESOLUÇÃO DINÂMICA DE ARQUIVOS DE ÁUDIO
# =============================================================================
def resolver_caminho_audio(caminho_fornecido: str | None, diretorio_base: str) -> str:
    """
    Valida o caminho fornecido via CLI ou resolve o arquivo de áudio padrão
    na pasta 'audios/' do projeto.

    Parâmetros:
        caminho_fornecido (str | None): Caminho passado por linha de comando.
        diretorio_base (str): Diretório base do repositório.

    Retorna:
        str: Caminho absoluto para o arquivo de áudio selecionado.
    """
    if caminho_fornecido:
        if os.path.isabs(caminho_fornecido):
            caminho_resolvido = caminho_fornecido
        else:
            caminho_resolvido = os.path.abspath(os.path.join(diretorio_base, caminho_fornecido))

        if not os.path.exists(caminho_resolvido):
            raise FileNotFoundError(f"O arquivo de áudio especificado não existe: {caminho_resolvido}")
        return caminho_resolvido

    pasta_audios = os.path.join(diretorio_base, 'audios')
    caminho_padrao = os.path.join(pasta_audios, 'som1.wav')

    if os.path.exists(caminho_padrao):
        return caminho_padrao

    # Se som1.wav não existir, busca qualquer .wav ou .mp3 na pasta audios/
    padroes = [os.path.join(pasta_audios, '*.wav'), os.path.join(pasta_audios, '*.mp3')]
    candidatos = []
    for padrao in padroes:
        candidatos.extend(glob.glob(padrao))

    if candidatos:
        candidatos.sort()
        return candidatos[0]

    raise FileNotFoundError(
        f"Nenhum arquivo de áudio encontrado na pasta padrão '{pasta_audios}'. "
        "Especifique o arquivo desejado como argumento na linha de comando."
    )


# =============================================================================
# BLOCO PRINCIPAL (CLI)
# =============================================================================
def main():
    """
    Ponto de entrada via linha de comando para execução do pipeline de transcrição.
    """
    diretorio_base = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline de Transcrição de Guitarra para Tablatura (TCC).\n"
            "Detecta ataques transitórios (onsets) e rastreia pitch fundamental (f0)\n"
            "diretamente no sinal de áudio bruto usando librosa.onset e pYIN."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'audio',
        nargs='?',
        default=None,
        help="Caminho relativo ou absoluto do arquivo de áudio (.wav ou .mp3). Padrão: audios/som1.wav."
    )
    parser.add_argument(
        '--visualizar',
        action='store_true',
        help="Gera e salva o espectrograma CQT com marcadores de onsets na pasta 'resultados/'."
    )
    parser.add_argument(
        '--saida',
        type=str,
        default=None,
        help="Caminho personalizado para salvar a imagem de visualização CQT (usado com --visualizar)."
    )

    args = parser.parse_args()

    try:
        caminho_audio = resolver_caminho_audio(args.audio, diretorio_base)
    except FileNotFoundError as err:
        print(f"[ERRO] {err}", file=sys.stderr)
        sys.exit(1)

    print("=" * 88)
    print(" PIPELINE DE TRANSCRIÇÃO MONOFÔNICA DE GUITARRA (TCC) ")
    print("=" * 88)
    print(f"Arquivo alvo: {caminho_audio}")

    # Executa a transcrição completa
    notas_detectadas = transcrever_audio(caminho_audio, sr=22050)

    # Imprime a tabela formatada no console
    imprimir_tabela_notas(notas_detectadas, nome_arquivo=os.path.basename(caminho_audio))

    # Geração opcional de imagem CQT
    if args.visualizar:
        print("[VISUALIZAÇÃO] Gerando espectrograma CQT com marcações de onsets...")
        pasta_resultados = os.path.join(diretorio_base, 'resultados')
        caminho_img = gerar_visualizacao_cqt(
            caminho_audio=caminho_audio,
            caminho_saida=args.saida,
            pasta_resultados=pasta_resultados,
            sr=22050
        )
        print(f"[VISUALIZAÇÃO] Imagem salva com sucesso em:\n               {caminho_img}\n")

    return notas_detectadas


if __name__ == '__main__':
    main()
