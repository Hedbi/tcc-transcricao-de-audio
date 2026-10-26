# =============================================================================
# TCC - Transcrição Automática de Áudio de Guitarra para Tablatura
# Módulo de Avaliação Padronizada: GuitarSet Dataset & mir_eval
# =============================================================================
"""
Avaliação Quantitativa de Transcrição Musical com GuitarSet e mir_eval
---------------------------------------------------------------------
Este módulo integra o pipeline de transcrição monofônica de guitarra aos
arquivos de anotação no formato JAMS (.jams) do dataset acadêmico GuitarSet,
computando métricas padronizadas da literatura de Recuperação de Informação
Musical (MIR - Music Information Retrieval) através da biblioteca mir_eval.

FUNDAMENTAÇÃO ACADÊMICA:
------------------------
Na avaliação de sistemas de transcrição automática de música (AMT - Automatic
Music Transcription), a avaliação puramente qualitativa ou visual é insuficiente
para estabelecer rigor científico. Este módulo adota os protocolos oficiais
estabelecidos pela comunidade acadêmica internacional (MIREX / ISMIR):

1. GuitarSet (Xi et al., ISMIR 2018):
   O GuitarSet é o benchmark padrão ouro para transcrição de violão/guitarra.
   Ele consiste em 360 gravações acústicas de alta fidelidade sincronizadas
   com captadores hexafônicos individuais por corda, garantindo ground-truth
   preciso de início (onset), término (offset) e altura (pitch) para cada
   uma das 6 cordas do instrumento.

2. mir_eval (Raffel et al., ISMIR 2014):
   Biblioteca de referência padrão para cálculo de métricas padronizadas:
     - Precisão (Precision): fração das notas transcritas que são verdadeiras.
     - Revocação (Recall): fração das notas reais que foram identificadas.
     - F1-Score: média harmônica ponderada entre Precisão e Revocação.
     - Raw Pitch Accuracy (RPA): acurácia melódica frame-a-frame de f0 contínua.

3. Tolerâncias Padronizadas (MIREX AMT Task):
     - Onset: tolerância temporal máxima de ±50 ms (0.05 s).
     - Pitch: tolerância de afinação de ±50 cents (quarto de tom / 0.5 semitom).
     - Offset: tolerância de ±20% da duração da nota ou 50 ms.

ESTRUTURA ESPERADA DO DATASET:
------------------------------
g:/TCC/datasets/guitarset/
    ├── annotation/         (arquivos .jams com anotações por corda)
    └── audio_mono-mic/     (arquivos .wav gravados com microfone condensador)
         ou audio_hex-deconv/ (áudios hexafônicos desconvolvidos)

Autor: Desenvolvido para o Trabalho de Conclusão de Curso (TCC).
"""

import os
import sys
import glob
import argparse
import csv
import warnings
from typing import Dict, List, Tuple, Any, Optional

# Garante compatibilidade de saída em consoles Windows com diferentes codificações
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import librosa

# -----------------------------------------------------------------------------
# TRATAMENTO SEGURO DE DEPENDÊNCIAS DE MIR (jams e mir_eval)
# -----------------------------------------------------------------------------
# Caso o usuário execute o script em um ambiente Python recém-criado sem as
# dependências instaladas, o script não encerra de forma abrupta com traceback;
# em vez disso, exibe orientações claras de instalação conforme o padrão acadêmico.
try:
    import jams
    JAMS_DISPONIVEL = True
except ImportError:
    jams = None
    JAMS_DISPONIVEL = False

try:
    import mir_eval
    MIR_EVAL_DISPONIVEL = True
except ImportError:
    mir_eval = None
    MIR_EVAL_DISPONIVEL = False

# -----------------------------------------------------------------------------
# INTEGRAÇÃO COM O PIPELINE DE TRANSCRIÇÃO (pipeline_transcritor.py)
# -----------------------------------------------------------------------------
try:
    from pipeline_transcritor import transcrever_audio
    PIPELINE_DISPONIVEL = True
except ImportError:
    transcrever_audio = None
    PIPELINE_DISPONIVEL = False


# =============================================================================
# SEÇÃO 1: VERIFICAÇÃO DE AMBIENTE E AUXILIAR DO DATASET (SETUP HELPER)
# =============================================================================

def verificar_dependencias() -> bool:
    """
    Verifica se as bibliotecas 'jams' e 'mir_eval' estão instaladas.
    Se alguma estiver ausente, imprime instruções claras de instalação.

    Retorna:
        bool: True se todas as dependências estiverem presentes; False caso contrário.
    """
    faltantes = []
    if not JAMS_DISPONIVEL:
        faltantes.append("jams")
    if not MIR_EVAL_DISPONIVEL:
        faltantes.append("mir_eval")

    if faltantes:
        print("\n" + "=" * 78)
        print(" [ERRO DE DEPENDÊNCIA] Bibliotecas essenciais de MIR não encontradas!")
        print("=" * 78)
        print(" Para executar a avaliação quantitativa com o GuitarSet, é necessário")
        print(" instalar os seguintes pacotes Python:")
        for pkg in faltantes:
            print(f"   • {pkg}")
        print("\n Comando de instalação recomendado via terminal:")
        print(f"   pip install {' '.join(faltantes)}")
        print("=" * 78 + "\n")
        return False
    return True


def imprimir_instrucoes_guitarset(diretorio_sugerido: str = "g:/TCC/datasets/guitarset"):
    """
    Exibe um guia detalhado e fundamentado para download, organização e
    estruturação do GuitarSet no ambiente local do projeto de TCC.
    """
    caminho_abs = os.path.abspath(diretorio_sugerido)
    print("\n" + "=" * 78)
    print(" GUIA DE CONFIGURACAO DO DATASET GUITARSET PARA AVALIACAO DO TCC")
    print("=" * 78)
    print("""
SOBRE O DATASET:
  O GuitarSet e uma colecao academica aberta desenvolvida por Qingyang Xi et al.
  (ISMIR 2018 / Universidade de Nova Iorque - NYU MARL). Ele contem 360 trechos
  de violao acustico (~30s cada), cobrindo multiplos guitarristas, compassos e
  estilos (comping ritmico e solos melodicos), com gravacoes hexafonicas e
  microfonicas rigorosamente anotadas em formato JAMS.

COMO BAIXAR:
  1. Acesse o repositorio oficial no Zenodo:
     https://zenodo.org/records/3371780

  2. Baixe os seguintes arquivos essenciais:
     * annotation.tar.gz      (~15 MB)  -> Anotacoes em formato .jams
     * audio_mono-mic.tar.gz  (~6.5 GB) -> Audio gravado em microfone estudio
     (Opcional para analise de corda isolada: audio_hex-deconv.tar.gz)

ESTRUTURA DE DIRETORIOS ESPERADA:
  Descompacte os arquivos de modo a formar a seguinte arvore de pastas:

  {caminho}
      |-- annotation/
      |   |-- 00_BN1-129-Eb_comp.jams
      |   |-- 00_BN1-129-Eb_solo.jams
      |   \\-- ... (360 arquivos .jams)
      \\-- audio_mono-mic/
          |-- 00_BN1-129-Eb_comp_mic.wav
          |-- 00_BN1-129-Eb_solo_mic.wav
          \\-- ... (360 arquivos .wav)

DICA DE EXECUCAO:
  Para criar essa estrutura de diretorios automaticamente no seu sistema, execute:
      python avaliar_com_guitarset.py --criar-pastas
""".format(caminho=caminho_abs))
    print("=" * 78 + "\n")


def criar_estrutura_pastas(caminho_base: str = "g:/TCC/datasets/guitarset") -> bool:
    """
    Cria a hierarquia de diretórios recomendada para receber o GuitarSet
    e grava um arquivo README informativo no local.

    Parâmetros:
        caminho_base (str): Caminho raiz onde o dataset ficará armazenado.

    Retorna:
        bool: True se as pastas foram preparadas com sucesso.
    """
    pastas = [
        os.path.join(caminho_base, "annotation"),
        os.path.join(caminho_base, "audio_mono-mic"),
        os.path.join(caminho_base, "audio_hex-deconv")
    ]

    for p in pastas:
        os.makedirs(p, exist_ok=True)

    readme_path = os.path.join(caminho_base, "LEIA-ME_GUITARSET.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "DATASET GUITARSET - ESTRUTURA PARA AVALIACAO DE TCC\n"
            "==================================================\n\n"
            "1. Coloque os arquivos .jams descompactados dentro da pasta:\n"
            "   annotation/\n\n"
            "2. Coloque os arquivos .wav gravados em microfone dentro da pasta:\n"
            "   audio_mono-mic/\n\n"
            "3. (Opcional) Coloque audios hexafonicos em:\n"
            "   audio_hex-deconv/\n\n"
            "Download oficial: https://zenodo.org/records/3371780\n"
        )

    print(f"\n[SUCESSO] Estrutura de diretorios criada em: {os.path.abspath(caminho_base)}")
    print(f"  * {pastas[0]}")
    print(f"  * {pastas[1]}")
    print(f"  * {pastas[2]}")
    print(f"  * Guia gravado em: {readme_path}\n")
    return True


def verificar_estrutura_dataset(
    caminho_base: str,
    tipo_audio: str = "audio_mono-mic"
) -> Tuple[bool, int, int, str]:
    """
    Inspeciona o diretório do GuitarSet e verifica a existência de anotações
    e arquivos de áudio correspondentes.

    Retorna:
        Tuple[bool, int, int, str]:
            - valido (bool): True se pastas existem e têm arquivos.
            - total_jams (int): Quantidade de arquivos .jams encontrados.
            - total_audios (int): Quantidade de arquivos .wav encontrados.
            - mensagem (str): Diagnóstico textual do estado do diretório.
    """
    if not os.path.exists(caminho_base):
        return False, 0, 0, f"Diretório raiz '{caminho_base}' não encontrado."

    pasta_annot = os.path.join(caminho_base, "annotation")
    pasta_audio = os.path.join(caminho_base, tipo_audio)

    if not os.path.exists(pasta_annot):
        return False, 0, 0, f"Pasta de anotações não encontrada: '{pasta_annot}'"

    if not os.path.exists(pasta_audio):
        return False, 0, 0, f"Pasta de áudio não encontrada: '{pasta_audio}'"

    jams_files = glob.glob(os.path.join(pasta_annot, "*.jams"))
    audio_files = glob.glob(os.path.join(pasta_audio, "*.wav"))

    if len(jams_files) == 0:
        return False, 0, len(audio_files), f"Nenhum arquivo .jams encontrado em '{pasta_annot}'"

    if len(audio_files) == 0:
        return False, len(jams_files), 0, f"Nenhum arquivo .wav encontrado em '{pasta_audio}'"

    return True, len(jams_files), len(audio_files), "Dataset encontrado e operacional."


# =============================================================================
# SEÇÃO 2: LEITURA E EXTRAÇÃO DAS ANOTAÇÕES JAMS (GUITARSET)
# =============================================================================

def extrair_notas_guitarset(
    caminho_jams: str,
    namespace_preferido: str = "note_midi",
    monofonico: bool = True
) -> List[Dict[str, Any]]:
    """
    Carrega o arquivo .jams de uma faixa do GuitarSet e extrai todas as notas
    musicais anotadas das 6 cordas do instrumento.

    Parâmetros:
        caminho_jams (str): Caminho para o arquivo .jams do GuitarSet.
        namespace_preferido (str): Namespace padrão para notas ('note_midi' ou 'pitch_midi').

    Retorna:
        List[Dict[str, Any]]: Lista de dicionários representando as notas de referência,
                              ordenada cronologicamente pelo instante de início (onset):
            - 'onset' (float): Instante inicial em segundos.
            - 'offset' (float): Instante final em segundos.
            - 'duracao' (float): Duração total em segundos.
            - 'midi' (float): Altura em semitons MIDI contínuos.
            - 'freq_hz' (float): Frequência fundamental de referência em Hertz.
            - 'nota' (str): Notação científica (ex: 'E2', 'A4').
            - 'corda' (int): Índice da corda da guitarra (0 = E2 / 6ª corda a 5 = E4 / 1ª corda).
            - 'confianca' (float | None): Confiança da anotação no arquivo JAMS.
        monofonico: Quando True, mantém uma única voz sem sobreposição temporal,
                   que é o escopo experimental atual do TCC.

    Justificativa Técnica (Estrutura Hexafônica do GuitarSet):
        Diferente de instrumentos monofônicos puros (como clarinete ou voz), o
        violão de 6 cordas possui 6 canais acústicos independentes que podem vibrar
        concomitantemente. No GuitarSet, o arquivo .jams armazena 6 anotações
        distintas sob o namespace 'note_midi' — uma para cada corda. Esta função
        varre as 6 anotações, extrai os eventos observados, mapeia os valores MIDI
        em frequências reais de referência em Hz e consolida uma lista cronológica
        de eventos sonoros.
    """
    if not JAMS_DISPONIVEL:
        raise RuntimeError("A biblioteca 'jams' é necessária para ler arquivos .jams.")

    if not os.path.exists(caminho_jams):
        raise FileNotFoundError(f"Arquivo JAMS não encontrado: {caminho_jams}")

    # Carregamento com validação desativada para máxima compatibilidade entre versões de schema
    jam = jams.load(caminho_jams, validate=False)

    # Busca anotações de notas discretas
    anotacoes = jam.search(namespace=namespace_preferido)
    if not anotacoes:
        # Fallback para o namespace secundário
        alt_ns = "pitch_midi" if namespace_preferido == "note_midi" else "note_midi"
        anotacoes = jam.search(namespace=alt_ns)

    if not anotacoes:
        warnings.warn(
            f"Nenhuma anotação de nota encontrada nos namespaces 'note_midi' ou 'pitch_midi' em {caminho_jams}",
            UserWarning
        )
        return []

    notas_extraidas: List[Dict[str, Any]] = []

    for ann_idx, ann in enumerate(anotacoes):
        # Determina o índice da corda (0 a 5)
        corda_idx = ann_idx
        if hasattr(ann, 'annotation_metadata') and ann.annotation_metadata.data_source:
            try:
                corda_idx = int(ann.annotation_metadata.data_source)
            except (ValueError, TypeError):
                corda_idx = ann_idx

        # Itera sobre todas as observações registradas na corda
        for obs in ann.data:
            t_onset = float(obs.time)
            t_dur = float(obs.duration)
            t_offset = t_onset + t_dur

            # O valor em 'note_midi' pode ser um float/int ou um dicionário {'value': ...}
            val = obs.value
            if isinstance(val, dict):
                midi_val = float(val.get('value', 0.0))
            else:
                try:
                    midi_val = float(val)
                except (ValueError, TypeError):
                    continue

            # Conversão para frequência exata (Hz) e notação alfanumérica
            freq_hz = float(librosa.midi_to_hz(midi_val))
            nota_nome = librosa.midi_to_note(int(round(midi_val)), unicode=False)

            registro = {
                'onset': round(t_onset, 4),
                'offset': round(t_offset, 4),
                'duracao': round(t_dur, 4),
                'midi': round(midi_val, 2),
                'freq_hz': round(freq_hz, 2),
                'nota': nota_nome,
                'corda': corda_idx,
                'confianca': obs.confidence if hasattr(obs, 'confidence') else None
            }
            notas_extraidas.append(registro)

    # Ordenação cronológica por tempo de início e término
    notas_extraidas.sort(key=lambda n: (n['onset'], n['offset']))
    if monofonico:
        return selecionar_voz_monofonica(notas_extraidas)
    return notas_extraidas


def selecionar_voz_monofonica(
    notas: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Seleciona uma voz não sobreposta para o experimento monofônico.

    O GuitarSet também contém faixas de acompanhamento e anotações de várias
    cordas. Em caso de sobreposição, a anotação com maior confiança é mantida;
    na ausência de confiança, a nota mais longa é usada como desempate
    determinístico. Isso torna explícita a redução de escopo, em vez de
    comparar uma saída monofônica com um acorde polifônico.
    """
    selecionadas: List[Dict[str, Any]] = []
    for nota in notas:
        if not selecionadas or nota['onset'] >= selecionadas[-1]['offset']:
            selecionadas.append(nota)
            continue

        anterior = selecionadas[-1]
        conf_anterior = anterior.get('confianca')
        conf_nota = nota.get('confianca')
        if conf_anterior is not None and conf_nota is not None:
            manter_nota = conf_nota > conf_anterior
        else:
            manter_nota = nota['offset'] - nota['onset'] > anterior['offset'] - anterior['onset']

        if manter_nota:
            selecionadas[-1] = nota

    return selecionadas


# =============================================================================
# SEÇÃO 3: INTEGRAÇÃO E EXECUÇÃO DO PIPELINE DE TRANSCRIÇÃO
# =============================================================================

def transcrever_audio_guitarset(caminho_audio: str, sr: int = 22050) -> List[Dict[str, Any]]:
    """
    Executa o pipeline de transcrição monofônica sobre o arquivo de áudio
    especificado e formata o resultado para avaliação.

    Parâmetros:
        caminho_audio (str): Caminho para o arquivo .wav.
        sr (int): Taxa de amostragem (padrão: 22050 Hz).

    Retorna:
        List[Dict[str, Any]]: Lista de notas estimadas com instantes e frequências.
    """
    if not PIPELINE_DISPONIVEL:
        raise ImportError(
            "O módulo 'pipeline_transcritor.py' não foi encontrado no PYTHONPATH ou diretório corrente.\n"
            "Certifique-se de que o arquivo pipeline_transcritor.py está na raiz do projeto (g:/TCC)."
        )

    if not os.path.exists(caminho_audio):
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {caminho_audio}")

    # Executa a transcrição via librosa.onset_detect + pYIN
    predicoes_brutas = transcrever_audio(caminho_audio, sr=sr)

    # Normalização e validação das chaves retornadas
    notas_preditas: List[Dict[str, Any]] = []
    for item in predicoes_brutas:
        # Obtenção de instantes e atributos com suporte a aliases
        t_onset = float(item.get('onset', item.get('onset_time', 0.0)))
        t_offset = float(item.get('offset', item.get('offset_time', t_onset + 0.1)))
        freq = item.get('freq_hz')

        # Se pYIN não identificou frequência válida (segmento surdo/ruído), descarta para AMT de notas
        if freq is None or (isinstance(freq, float) and (np.isnan(freq) or freq <= 0)):
            continue

        midi_val = item.get('midi', item.get('midi_number'))
        if midi_val is None:
            midi_val = float(librosa.hz_to_midi(freq))

        nome_nota = item.get('nota', item.get('note_name', ''))
        if not nome_nota or nome_nota == 'N/A':
            nome_nota = librosa.midi_to_note(int(round(midi_val)), unicode=False)

        notas_preditas.append({
            'onset': round(t_onset, 4),
            'offset': round(t_offset, 4),
            'duracao': round(t_offset - t_onset, 4),
            'freq_hz': round(float(freq), 2),
            'midi': round(float(midi_val), 2),
            'nota': nome_nota
        })

    notas_preditas.sort(key=lambda n: (n['onset'], n['offset']))
    return notas_preditas


# =============================================================================
# SEÇÃO 4: CONVERSÃO PARA SÉRIE TEMPORAL E MÉTRICAS MELÓDICAS CONTÍNUAS (RPA)
# =============================================================================

def notas_para_serie_f0(
    lista_notas: List[Dict[str, Any]],
    duracao_maxima: float,
    hop_s: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converte uma lista discreta de eventos de notas (onset, offset, freq_hz)
    em uma série temporal contínua e uniforme de f0 (100 Hz / frames de 10 ms),
    necessária para o cálculo da métrica mir_eval.melody.raw_pitch_accuracy.

    Parâmetros:
        lista_notas: Lista de dicionários de notas.
        duracao_maxima: Duração total do áudio em segundos.
        hop_s: Resolução temporal em segundos (padrão 0.01 s = 10 ms).

    Retorna:
        Tuple[np.ndarray, np.ndarray]: Vetores alinhados de instantes de tempo e f0 em Hz.
    """
    tempos = np.arange(0.0, duracao_maxima + hop_s, hop_s)
    f0 = np.zeros_like(tempos)

    for nota in lista_notas:
        f = nota.get('freq_hz')
        if f is None or f <= 0:
            continue
        t0 = nota['onset']
        t1 = nota['offset']
        # Identifica os índices dentro da janela da nota
        idx = (tempos >= t0) & (tempos <= t1)
        f0[idx] = float(f)

    return tempos, f0


# =============================================================================
# SEÇÃO 5: CÁLCULO DE MÉTRICAS mir_eval E ANÁLISE DE DISCREPÂNCIAS
# =============================================================================

def avaliar_transcricao(
    ref_notas: List[Dict[str, Any]],
    est_notas: List[Dict[str, Any]],
    duracao_audio: Optional[float] = None,
    onset_tolerance: float = 0.05,
    pitch_tolerance: float = 50.0,
    offset_ratio: float = 0.2,
    offset_min_tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Computa o conjunto abrangente de métricas padronizadas mir_eval comparando
    as predições do pipeline com as anotações ground-truth do GuitarSet.

    Parâmetros:
        ref_notas: Lista de notas de referência (Ground-Truth do GuitarSet).
        est_notas: Lista de notas estimadas pelo pipeline_transcritor.
        duracao_audio: Duração total em segundos (para séries temporais de melodia).
        onset_tolerance: Tolerância temporal de ataque em segundos (padrão: 0.05 s = 50 ms).
        pitch_tolerance: Tolerância de afinação em cents (padrão: 50 cents = quarto de tom).
        offset_ratio: Proporção de tolerância de sustentação (padrão: 0.20 = 20%).
        offset_min_tolerance: Tolerância mínima absoluta de sustentação (padrão: 0.05 s = 50 ms).

    Retorna:
        Dict[str, Any]: Dicionário consolidado contendo:
            - Contagens (ref, est, true_positives, false_positives, false_negatives)
            - Métricas completas com offset (Precision, Recall, F1, Overlap)
            - Métricas sem offset (Precision, Recall, F1)
            - Métricas de onset temporal puro (Precision, Recall, F1)
            - Métrica de acurácia melódica contínua (Raw Pitch Accuracy - RPA)
            - Análise de discrepâncias detalhadas (Erros de oitava, desvios de afinação)
    """
    if not MIR_EVAL_DISPONIVEL:
        raise RuntimeError("A biblioteca 'mir_eval' é necessária para computar as métricas.")

    # 1. Preparação das matrizes NumPy nos formatos exigidos pelo mir_eval
    if len(ref_notas) > 0:
        ref_intervals = np.array([[n['onset'], n['offset']] for n in ref_notas], dtype=float)
        ref_pitches = np.array([n['freq_hz'] for n in ref_notas], dtype=float)
    else:
        ref_intervals = np.empty((0, 2), dtype=float)
        ref_pitches = np.empty((0,), dtype=float)

    if len(est_notas) > 0:
        est_intervals = np.array([[n['onset'], n['offset']] for n in est_notas], dtype=float)
        est_pitches = np.array([n['freq_hz'] for n in est_notas], dtype=float)
    else:
        est_intervals = np.empty((0, 2), dtype=float)
        est_pitches = np.empty((0,), dtype=float)

    total_ref = len(ref_notas)
    total_est = len(est_notas)

    # 2. Avaliação de Transcrição Completa: Onset + Pitch + Offset (Critério Estrito)
    # Suprime avisos do mir_eval no caso de matrizes vazias para manter console limpo
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prec_estrito, rec_estrito, f1_estrito, avg_overlap = (
            mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals,
                ref_pitches,
                est_intervals,
                est_pitches,
                onset_tolerance=onset_tolerance,
                pitch_tolerance=pitch_tolerance,
                offset_ratio=offset_ratio,
                offset_min_tolerance=offset_min_tolerance
            )
        )

        # 3. Avaliação sem restrição de Offset (Critério Onset + Pitch)
        # Fundamental para instrumentos com decaimento exponencial livre como violão/guitarra
        prec_no_off, rec_no_off, f1_no_off, _ = (
            mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals,
                ref_pitches,
                est_intervals,
                est_pitches,
                onset_tolerance=onset_tolerance,
                pitch_tolerance=pitch_tolerance,
                offset_ratio=None
            )
        )

        # 4. Avaliação Temporal Isolada (Apenas Onset)
        # Mede a eficácia isolada do Spectral Flux / onset_detect
        prec_ons, rec_ons, f1_ons = (
            mir_eval.transcription.onset_precision_recall_f1(
                ref_intervals,
                est_intervals,
                onset_tolerance=onset_tolerance
            )
        )

    # 5. Análise de Casamento e Erros Específicos (Mismatches)
    # Identifica exatamente quais notas foram correlacionadas
    pares_casados = []
    if total_ref > 0 and total_est > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Casamento com base em Onset + Pitch
            pares_casados = mir_eval.transcription.match_notes(
                ref_intervals,
                ref_pitches,
                est_intervals,
                est_pitches,
                onset_tolerance=onset_tolerance,
                pitch_tolerance=pitch_tolerance,
                offset_ratio=None
            )

    idx_ref_casados = {int(r) for r, e in pares_casados}
    idx_est_casados = {int(e) for r, e in pares_casados}

    # Falsos Negativos: Notas reais do Ground-Truth omitidas pelo algoritmo
    falsos_negativos = [ref_notas[i] for i in range(total_ref) if i not in idx_ref_casados]

    # Falsos Positivos: Notas preditas espúrias (ruído, batidas ou harmônicos dissociados)
    falsos_positivos = [est_notas[i] for i in range(total_est) if i not in idx_est_casados]

    # 6. Identificação Específica de Erros de Oitava (Octave Errors)
    # Ocorre quando o onset é detectado com exatidão temporal, mas o pitch é dobrado ou cortado à metade
    erros_oitava = []
    desvios_pitch = []

    if total_ref > 0 and total_est > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pares_onset = mir_eval.transcription.match_note_onsets(
                ref_intervals,
                est_intervals,
                onset_tolerance=onset_tolerance
            )

        for r, e in pares_onset:
            r_idx, e_idx = int(r), int(e)
            r_nota = ref_notas[r_idx]
            e_nota = est_notas[e_idx]

            diff_midi = abs(r_nota['midi'] - e_nota['midi'])
            diff_cents = 1200.0 * abs(np.log2(e_nota['freq_hz'] / r_nota['freq_hz']))

            # Se o desvio excede a tolerância de 50 cents
            if diff_cents > pitch_tolerance:
                # Verifica erro característico de oitava (12 semitons ± 1)
                is_oitava = abs(diff_midi - 12.0) <= 0.5 or abs(diff_midi - 24.0) <= 0.5
                erro_item = {
                    'tempo_onset_ref': r_nota['onset'],
                    'tempo_onset_est': e_nota['onset'],
                    'ref_nota': r_nota['nota'],
                    'ref_freq': r_nota['freq_hz'],
                    'ref_corda': r_nota.get('corda', 'N/A'),
                    'est_nota': e_nota['nota'],
                    'est_freq': e_nota['freq_hz'],
                    'diff_cents': round(diff_cents, 1),
                    'diff_semitons': round(diff_midi, 1)
                }
                if is_oitava:
                    erros_oitava.append(erro_item)
                else:
                    desvios_pitch.append(erro_item)

    # 7. Acurácia Melódica Contínua Frame a Frame (Raw Pitch Accuracy - RPA)
    rpa = 0.0
    rca = 0.0
    oa = 0.0
    if duracao_audio is None:
        duracoes = [n['offset'] for n in (ref_notas + est_notas)]
        duracao_audio = max(duracoes) if duracoes else 1.0

    if total_ref > 0 and total_est > 0:
        try:
            t_ref, f_ref = notas_para_serie_f0(ref_notas, duracao_audio, hop_s=0.01)
            t_est, f_est = notas_para_serie_f0(est_notas, duracao_audio, hop_s=0.01)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(
                    t_ref, f_ref, t_est, f_est
                )
                rpa = float(mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=pitch_tolerance))
                rca = float(mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=pitch_tolerance))
                oa = float(mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=pitch_tolerance))
        except Exception:
            rpa, rca, oa = 0.0, 0.0, 0.0

    return {
        'total_referencia': total_ref,
        'total_preditas': total_est,
        'verdadeiros_positivos': len(pares_casados),
        'falsos_positivos': len(falsos_positivos),
        'falsos_negativos': len(falsos_negativos),
        # Critério Completo (Com Offset)
        'precisao_estrita': float(prec_estrito),
        'revocacao_estrita': float(rec_estrito),
        'f1_estrito': float(f1_estrito),
        'overlap_medio': float(avg_overlap),
        # Critério Onset + Pitch (Sem Offset)
        'precisao_nota': float(prec_no_off),
        'revocacao_nota': float(rec_no_off),
        'f1_nota': float(f1_no_off),
        # Critério Onset Puro
        'precisao_onset': float(prec_ons),
        'revocacao_onset': float(rec_ons),
        'f1_onset': float(f1_ons),
        # Métricas Melódicas Contínuas
        'raw_pitch_accuracy': float(rpa),
        'raw_chroma_accuracy': float(rca),
        'overall_accuracy': float(oa),
        # Diagnóstico Qualitativo de Erros
        'total_erros_oitava': len(erros_oitava),
        'erros_oitava': erros_oitava,
        'desvios_pitch': desvios_pitch,
        'amostra_falsos_negativos': falsos_negativos[:5],
        'amostra_falsos_positivos': falsos_positivos[:5],
        'lista_completa_falsos_negativos': falsos_negativos,
        'lista_completa_falsos_positivos': falsos_positivos
    }


# =============================================================================
# SEÇÃO 6: LOCALIZAÇÃO E PAREAMENTO DE ARQUIVOS NO GUITARSET
# =============================================================================

def normalizar_id_arquivo(nome_arquivo: str) -> str:
    """
    Remove extensões e sufixos instrumentais/microfônicos do nome de arquivo
    do GuitarSet para permitir pareamento biunívoco entre anotação e áudio.

    Exemplos:
        '00_BN1-129-Eb_solo_mic.wav'  -> '00_BN1-129-Eb_solo'
        '00_BN1-129-Eb_solo.jams'     -> '00_BN1-129-Eb_solo'
        '01_SS2-88-F_comp_hex_cln.wav' -> '01_SS2-88-F_comp'
    """
    base = os.path.splitext(os.path.basename(nome_arquivo))[0]
    for suf in ["_mic", "_hex_cln", "_hex", "_pickup", "_clean"]:
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    return base


def localizar_pares_guitarset(
    diretorio_base: str,
    tipo_audio: str = "audio_mono-mic",
    filtro_estilo: Optional[str] = None
) -> List[Tuple[str, str, str]]:
    """
    Varre o dataset GuitarSet e encontra todos os pares correlacionados
    entre arquivos de áudio (.wav) e anotações JAMS (.jams).

    Parâmetros:
        diretorio_base (str): Caminho raiz do GuitarSet.
        tipo_audio (str): Subpasta de áudio ('audio_mono-mic' ou 'audio_hex-deconv').
        filtro_estilo (str, opcional): Filtrar por 'solo', 'comp' ou None (todos).

    Retorna:
        List[Tuple[str, str, str]]: Lista de tuplas (caminho_audio, caminho_jams, id_base).
    """
    pasta_annot = os.path.join(diretorio_base, "annotation")
    pasta_audio = os.path.join(diretorio_base, tipo_audio)

    if not os.path.isdir(pasta_annot) or not os.path.isdir(pasta_audio):
        return []

    jams_map: Dict[str, str] = {}
    for p in glob.glob(os.path.join(pasta_annot, "*.jams")):
        id_b = normalizar_id_arquivo(p)
        jams_map[id_b] = p

    pares: List[Tuple[str, str, str]] = []
    for p_audio in glob.glob(os.path.join(pasta_audio, "*.wav")):
        id_b = normalizar_id_arquivo(p_audio)

        # Aplica filtro por estilo musical (solo vs comping)
        if filtro_estilo:
            filtro_limpo = filtro_estilo.strip().lower()
            if filtro_limpo in ["solo", "comp"]:
                if f"_{filtro_limpo}" not in id_b.lower():
                    continue

        if id_b in jams_map:
            pares.append((p_audio, jams_map[id_b], id_b))

    pares.sort(key=lambda x: x[2])
    return pares


# =============================================================================
# SEÇÃO 7: APRESENTAÇÃO DE RELATÓRIOS E EXPORTAÇÃO CSV
# =============================================================================

def imprimir_relatorio_individual(
    res: Dict[str, Any],
    caminho_audio: str,
    caminho_jams: str,
    detalhado: bool = False
):
    """
    Exibe no console o relatório formatado com as métricas mir_eval e diagnóstico
    de erros para um arquivo individual avaliado.
    """
    nome_audio = os.path.basename(caminho_audio)
    nome_jams = os.path.basename(caminho_jams)

    print("\n" + "=" * 80)
    print(" RELATÓRIO DE AVALIAÇÃO COM GUITARSET (mir_eval)")
    print("=" * 80)
    print(f" Arquivo de Áudio:     {nome_audio}")
    print(f" Arquivo de Anotação:  {nome_jams}")
    print("-" * 80)
    print(" CONTAGEM DE EVENTOS NOTA-A-NOTA:")
    print(f"   • Notas de Referência (Ground-Truth): {res['total_referencia']}")
    print(f"   • Notas Preditas pelo Pipeline:      {res['total_preditas']}")
    print(f"   • Notas Corretas (True Positives):   {res['verdadeiros_positivos']}")
    print(f"   • Falsos Positivos (Notas Espúrias): {res['falsos_positivos']}")
    print(f"   • Falsos Negativos (Notas Omitidas): {res['falsos_negativos']}")
    print("-" * 80)
    print(" MÉTRICAS PADRONIZADAS (MIREX / ISMIR):")
    print("   [1] Critério Onset + Pitch (Tolerância: Onset ±50ms, Pitch ±50 cents):")
    print(f"       • Precisão (Precision): {res['precisao_nota']:.4f} ({res['precisao_nota']*100:.2f}%)")
    print(f"       • Revocação (Recall):   {res['revocacao_nota']:.4f} ({res['revocacao_nota']*100:.2f}%)")
    print(f"       • F1-Score:             {res['f1_nota']:.4f} ({res['f1_nota']*100:.2f}%)")

    print("\n   [2] Critério Estrito Completo (Com Tolerância de Offset ±20%):")
    print(f"       • Precisão (Precision): {res['precisao_estrita']:.4f} ({res['precisao_estrita']*100:.2f}%)")
    print(f"       • Revocação (Recall):   {res['revocacao_estrita']:.4f} ({res['revocacao_estrita']*100:.2f}%)")
    print(f"       • F1-Score:             {res['f1_estrito']:.4f} ({res['f1_estrito']*100:.2f}%)")
    print(f"       • Sobreposição Média:   {res['overlap_medio']:.4f}")

    print("\n   [3] Detecção Temporal Isolada (Apenas Onset ±50ms):")
    print(f"       • Precisão de Ataque:   {res['precisao_onset']:.4f} ({res['precisao_onset']*100:.2f}%)")
    print(f"       • Revocação de Ataque:  {res['revocacao_onset']:.4f} ({res['revocacao_onset']*100:.2f}%)")
    print(f"       • F1-Score de Ataque:   {res['f1_onset']:.4f} ({res['f1_onset']*100:.2f}%)")

    print("\n   [4] Rastreamento Melódico Contínuo (Série Temporal frame-a-frame):")
    print(f"       • Raw Pitch Accuracy (RPA):  {res['raw_pitch_accuracy']:.4f} ({res['raw_pitch_accuracy']*100:.2f}%)")
    print(f"       • Raw Chroma Accuracy (RCA): {res['raw_chroma_accuracy']:.4f} ({res['raw_chroma_accuracy']*100:.2f}%)")
    print(f"       • Overall Accuracy (OA):     {res['overall_accuracy']:.4f} ({res['overall_accuracy']*100:.2f}%)")

    print("-" * 80)
    print(" ANÁLISE QUALITATIVA DE DISCREPÂNCIAS E ERROS:")
    print(f"   • Erros de Oitava Detectados: {res['total_erros_oitava']}")
    if res['total_erros_oitava'] > 0:
        for err in res['erros_oitava'][:4]:
            print(f"     - t={err['tempo_onset_ref']:.2f}s: Ref {err['ref_nota']} ({err['ref_freq']} Hz, C{err['ref_corda']}) vs "
                  f"Pred {err['est_nota']} ({err['est_freq']} Hz) [Δ={err['diff_semitons']} semitons]")
        if res['total_erros_oitava'] > 4 and not detalhado:
            print(f"     ... e mais {res['total_erros_oitava'] - 4} erros de oitava (use --detalhado para ver todos).")

    if res['desvios_pitch']:
        print(f"   • Outros Desvios de Afinação: {len(res['desvios_pitch'])}")
        for err in res['desvios_pitch'][:3]:
            print(f"     - t={err['tempo_onset_ref']:.2f}s: Ref {err['ref_nota']} ({err['ref_freq']} Hz) vs "
                  f"Pred {err['est_nota']} ({err['est_freq']} Hz) [Δ={err['diff_cents']} cents]")

    if res['amostra_falsos_negativos']:
        print(f"   • Amostra de Falsos Negativos (Notas reais não detectadas):")
        for fn in res['amostra_falsos_negativos'][:3]:
            print(f"     - Nota {fn['nota']} ({fn['freq_hz']} Hz, Corda {fn.get('corda','?')}) em [{fn['onset']:.2f}s - {fn['offset']:.2f}s]")

    if res['amostra_falsos_positivos']:
        print(f"   • Amostra de Falsos Positivos (Notas preditas espúrias):")
        for fp in res['amostra_falsos_positivos'][:3]:
            print(f"     - Nota {fp['nota']} ({fp['freq_hz']} Hz) em [{fp['onset']:.2f}s - {fp['offset']:.2f}s]")

    print("=" * 80 + "\n")


def imprimir_relatorio_lote(lista_resultados: List[Dict[str, Any]]):
    """
    Exibe a tabela consolidada de resultados para a avaliação em lote (batch),
    incluindo Médias Macro (por faixa) e Médias Micro (ponderadas pelo total de notas).
    """
    total_arquivos = len(lista_resultados)
    if total_arquivos == 0:
        print("\n[AVISO] Nenhum resultado disponível para exibição em lote.\n")
        return

    print("\n" + "=" * 90)
    print(" RESUMO DA AVALIAÇÃO EM LOTE (BATCH) - GUITARSET")
    print(f" Total de Faixas Avaliadas: {total_arquivos}")
    print("=" * 90)
    print(f"{'Faixa (ID)':<30} | {'Ref':<5} | {'Pred':<5} | {'Prec(O+P)':<9} | {'Rec(O+P)':<9} | {'F1(O+P)':<9} | {'F1(Est)':<9} | {'RPA':<7}")
    print("-" * 90)

    for item in lista_resultados:
        id_f = item['id_faixa'][:28]
        m = item['metricas']
        print(f"{id_f:<30} | {m['total_referencia']:<5} | {m['total_preditas']:<5} | "
              f"{m['precisao_nota']:<9.4f} | {m['revocacao_nota']:<9.4f} | {m['f1_nota']:<9.4f} | "
              f"{m['f1_estrito']:<9.4f} | {m['raw_pitch_accuracy']:<7.4f}")

    print("-" * 90)

    # Cálculo de Médias Macro (média simples sobre os arquivos)
    media_f1_op = np.mean([x['metricas']['f1_nota'] for x in lista_resultados])
    media_prec_op = np.mean([x['metricas']['precisao_nota'] for x in lista_resultados])
    media_rec_op = np.mean([x['metricas']['revocacao_nota'] for x in lista_resultados])
    media_f1_est = np.mean([x['metricas']['f1_estrito'] for x in lista_resultados])
    media_f1_ons = np.mean([x['metricas']['f1_onset'] for x in lista_resultados])
    media_rpa = np.mean([x['metricas']['raw_pitch_accuracy'] for x in lista_resultados])

    # Cálculo de Médias Micro (ponderadas pelo volume total de notas)
    soma_tp = sum(x['metricas']['verdadeiros_positivos'] for x in lista_resultados)
    soma_fp = sum(x['metricas']['falsos_positivos'] for x in lista_resultados)
    soma_fn = sum(x['metricas']['falsos_negativos'] for x in lista_resultados)

    prec_micro = soma_tp / (soma_tp + soma_fp) if (soma_tp + soma_fp) > 0 else 0.0
    rec_micro = soma_tp / (soma_tp + soma_fn) if (soma_tp + soma_fn) > 0 else 0.0
    f1_micro = (2 * prec_micro * rec_micro / (prec_micro + rec_micro)) if (prec_micro + rec_micro) > 0 else 0.0

    print(" MÉDIAS GERAIS:")
    print("   • Médias Macro (Aritmética entre as faixas):")
    print(f"       - F1-Score (Onset + Pitch):     {media_f1_op:.4f} ({media_f1_op*100:.2f}%)")
    print(f"       - Precisão Média (O+P):         {media_prec_op:.4f} ({media_prec_op*100:.2f}%)")
    print(f"       - Revocação Média (O+P):        {media_rec_op:.4f} ({media_rec_op*100:.2f}%)")
    print(f"       - F1-Score Estrito (c/ Offset): {media_f1_est:.4f} ({media_f1_est*100:.2f}%)")
    print(f"       - F1-Score Detecção de Onsets:  {media_f1_ons:.4f} ({media_f1_ons*100:.2f}%)")
    print(f"       - Raw Pitch Accuracy (RPA):     {media_rpa:.4f} ({media_rpa*100:.2f}%)")
    print("   • Médias Micro (Ponderada pelo volume total de notas do corpus):")
    print(f"       - Precisão Micro:               {prec_micro:.4f} ({prec_micro*100:.2f}%)")
    print(f"       - Revocação Micro:              {rec_micro:.4f} ({rec_micro*100:.2f}%)")
    print(f"       - F1-Score Micro:               {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    print("=" * 90 + "\n")


def exportar_relatorio_csv(lista_resultados: List[Dict[str, Any]], caminho_csv: str):
    """
    Exporta a série completa de métricas de avaliação para um arquivo CSV tabular,
    facilitando a elaboração de tabelas e gráficos comparativos na monografia do TCC.
    """
    os.makedirs(os.path.dirname(os.path.abspath(caminho_csv)), exist_ok=True)

    campos = [
        "id_faixa", "arquivo_audio", "arquivo_jams",
        "total_referencia", "total_preditas", "verdadeiros_positivos",
        "falsos_positivos", "falsos_negativos",
        "precisao_nota", "revocacao_nota", "f1_nota",
        "precisao_estrita", "revocacao_estrita", "f1_estrito", "overlap_medio",
        "precisao_onset", "revocacao_onset", "f1_onset",
        "raw_pitch_accuracy", "raw_chroma_accuracy", "overall_accuracy",
        "erros_oitava"
    ]

    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for item in lista_resultados:
            m = item['metricas']
            linha = {
                "id_faixa": item['id_faixa'],
                "arquivo_audio": os.path.basename(item['caminho_audio']),
                "arquivo_jams": os.path.basename(item['caminho_jams']),
                "total_referencia": m['total_referencia'],
                "total_preditas": m['total_preditas'],
                "verdadeiros_positivos": m['verdadeiros_positivos'],
                "falsos_positivos": m['falsos_positivos'],
                "falsos_negativos": m['falsos_negativos'],
                "precisao_nota": round(m['precisao_nota'], 4),
                "revocacao_nota": round(m['revocacao_nota'], 4),
                "f1_nota": round(m['f1_nota'], 4),
                "precisao_estrita": round(m['precisao_estrita'], 4),
                "revocacao_estrita": round(m['revocacao_estrita'], 4),
                "f1_estrito": round(m['f1_estrito'], 4),
                "overlap_medio": round(m['overlap_medio'], 4),
                "precisao_onset": round(m['precisao_onset'], 4),
                "revocacao_onset": round(m['revocacao_onset'], 4),
                "f1_onset": round(m['f1_onset'], 4),
                "raw_pitch_accuracy": round(m['raw_pitch_accuracy'], 4),
                "raw_chroma_accuracy": round(m['raw_chroma_accuracy'], 4),
                "overall_accuracy": round(m['overall_accuracy'], 4),
                "erros_oitava": m['total_erros_oitava']
            }
            writer.writerow(linha)

    print(f"[EXPORTAÇÃO] Relatório em CSV gravado com sucesso em: {os.path.abspath(caminho_csv)}")


# =============================================================================
# SEÇÃO 8: FLUXOS PRINCIPAIS DE EXECUÇÃO
# =============================================================================

def executar_avaliacao_individual(
    caminho_audio: str,
    caminho_jams: str,
    onset_tol: float = 0.05,
    pitch_tol: float = 50.0,
    offset_ratio: float = 0.2,
    detalhado: bool = False
) -> Dict[str, Any]:
    """
    Executa a avaliação completa de um único arquivo de áudio frente à sua
    anotação JAMS de referência.
    """
    print(f"\n[AVALIAÇÃO] Processando arquivo:")
    print(f"  • Áudio:    {caminho_audio}")
    print(f"  • Anotação: {caminho_jams}")

    # 1. Carrega as notas de referência do GuitarSet
    ref_notas = extrair_notas_guitarset(caminho_jams)
    print(f"  • Notas de Ground-Truth extraídas (6 cordas): {len(ref_notas)}")

    # 2. Executa o pipeline de transcrição
    est_notas = transcrever_audio_guitarset(caminho_audio)
    print(f"  • Notas identificadas pelo pipeline: {len(est_notas)}")

    # Duração do áudio para melodia contínua
    y, sr = librosa.load(caminho_audio, sr=22050)
    duracao_s = float(len(y) / sr)

    # 3. Computa métricas mir_eval
    metricas = avaliar_transcricao(
        ref_notas=ref_notas,
        est_notas=est_notas,
        duracao_audio=duracao_s,
        onset_tolerance=onset_tol,
        pitch_tolerance=pitch_tol,
        offset_ratio=offset_ratio
    )

    # 4. Apresenta o relatório
    imprimir_relatorio_individual(
        metricas,
        caminho_audio=caminho_audio,
        caminho_jams=caminho_jams,
        detalhado=detalhado
    )

    return metricas


def executar_avaliacao_lote(
    diretorio_dataset: str,
    tipo_audio: str = "audio_mono-mic",
    filtro_estilo: Optional[str] = "solo",
    limite_arquivos: Optional[int] = None,
    caminho_csv: Optional[str] = None,
    onset_tol: float = 0.05,
    pitch_tol: float = 50.0,
    offset_ratio: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Executa a avaliação automatizada em lote de múltiplos arquivos pareados
    do dataset GuitarSet.
    """
    pares = localizar_pares_guitarset(
        diretorio_base=diretorio_dataset,
        tipo_audio=tipo_audio,
        filtro_estilo=filtro_estilo
    )

    if not pares:
        print(f"\n[ERRO] Nenhum par coincidente de áudio e anotação JAMS foi encontrado em:")
        print(f"  {diretorio_dataset}")
        print("  Execute com a flag --setup para verificar as instruções de organização de pastas.")
        return []

    if limite_arquivos and limite_arquivos > 0:
        pares = pares[:limite_arquivos]

    print(f"\n" + "=" * 80)
    print(f" INICIANDO AVALIAÇÃO EM LOTE - GUITARSET")
    print(f" Total de pares selecionados: {len(pares)}")
    print(f" Diretório base:  {diretorio_dataset}")
    print(f" Tipo de áudio:   {tipo_audio}")
    print(f" Filtro estilo:   {filtro_estilo if filtro_estilo else 'Todos (solo e comp)'}")
    print("=" * 80)

    lista_resultados: List[Dict[str, Any]] = []

    for idx, (p_audio, p_jams, id_faixa) in enumerate(pares, start=1):
        print(f"\n[{idx}/{len(pares)}] Processando: {id_faixa} ...")
        try:
            ref_notas = extrair_notas_guitarset(p_jams)
            est_notas = transcrever_audio_guitarset(p_audio)

            # Estima duração com base nos eventos
            todas_notas = ref_notas + est_notas
            dur = max([n['offset'] for n in todas_notas]) if todas_notas else 10.0

            m = avaliar_transcricao(
                ref_notas=ref_notas,
                est_notas=est_notas,
                duracao_audio=dur,
                onset_tolerance=onset_tol,
                pitch_tolerance=pitch_tol,
                offset_ratio=offset_ratio
            )

            lista_resultados.append({
                'id_faixa': id_faixa,
                'caminho_audio': p_audio,
                'caminho_jams': p_jams,
                'metricas': m
            })
            print(f"    ✓ Concluído: F1(O+P) = {m['f1_nota']:.4f} | F1(Estrito) = {m['f1_estrito']:.4f} | RPA = {m['raw_pitch_accuracy']:.4f}")
        except Exception as e:
            print(f"    ✗ Falha no processamento de {id_faixa}: {e}")

    # Apresenta tabela resumo consolidada
    imprimir_relatorio_lote(lista_resultados)

    # Exporta para CSV se solicitado
    if caminho_csv:
        exportar_relatorio_csv(lista_resultados, caminho_csv)

    return lista_resultados


# =============================================================================
# BLOCO DE ENTRADA CLI (INTERFACE DE LINHA DE COMANDO)
# =============================================================================

def construir_parser() -> argparse.ArgumentParser:
    """
    Configura a interface de argumentos de linha de comando com argparse.
    """
    parser = argparse.ArgumentParser(
        description="Avaliação Padronizada de Transcrição Musical com GuitarSet e mir_eval (TCC)"
    )

    # Modos de operação principais
    parser.add_argument(
        "--jams",
        type=str,
        default=None,
        help="Caminho para um arquivo de anotação .jams específico a avaliar."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Caminho para o arquivo .wav correspondente (se omitido com --jams, tenta inferir)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "guitarset"),
        help="Diretório raiz do GuitarSet (padrão: datasets/guitarset)."
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Executa a avaliação em lote de todas as faixas encontradas no dataset."
    )
    parser.add_argument(
        "--audio-type",
        type=str,
        choices=["audio_mono-mic", "audio_hex-deconv"],
        default="audio_mono-mic",
        help="Subpasta de áudio do GuitarSet (padrão: audio_mono-mic)."
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["solo", "comp", "all"],
        default="solo",
        help="Filtro de estilo musical do GuitarSet ('solo' recomendado para monofônico, 'comp', ou 'all')."
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limite máximo de arquivos para processar na avaliação em lote."
    )
    parser.add_argument(
        "--salvar-csv",
        type=str,
        default=None,
        help="Caminho para salvar os resultados consolidados em formato CSV."
    )
    parser.add_argument(
        "--detalhado",
        action="store_true",
        help="Exibe todas as discrepâncias e notas no relatório individual."
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Exibe instruções acadêmicas de download e configuração do GuitarSet."
    )
    parser.add_argument(
        "--criar-pastas",
        action="store_true",
        help="Cria a estrutura de pastas necessária para receber o GuitarSet."
    )

    # Parâmetros de tolerância acústica do mir_eval
    parser.add_argument(
        "--tolerancia-onset",
        type=float,
        default=0.05,
        help="Tolerância temporal máxima de ataque em segundos (padrão MIREX: 0.05 s = 50 ms)."
    )
    parser.add_argument(
        "--tolerancia-pitch",
        type=float,
        default=50.0,
        help="Tolerância de afinação em cents (padrão MIREX: 50.0 cents = 0.5 semitom)."
    )
    parser.add_argument(
        "--tolerancia-offset",
        type=float,
        default=0.20,
        help="Proporção da duração para tolerância de offset (padrão MIREX: 0.20 = 20%%)."
    )

    return parser


def main():
    """
    Ponto de entrada do script de avaliação acadêmica.
    """
    parser = construir_parser()
    args = parser.parse_args()

    # 1. Modo Auxiliar de Instruções e Setup
    if args.setup:
        imprimir_instrucoes_guitarset(diretorio_sugerido=args.dataset_dir)
        sys.exit(0)

    # 2. Modo Criação Automática de Pastas
    if args.criar_pastas:
        criar_estrutura_pastas(caminho_base=args.dataset_dir)
        sys.exit(0)

    # 3. Verificação de Dependências Essenciais
    if not verificar_dependencias():
        sys.exit(1)

    # 4. Modo Avaliação de Arquivo Único
    if args.jams:
        caminho_jams = os.path.abspath(args.jams)
        caminho_audio = args.audio

        # Se o caminho de áudio não foi explicitamente fornecido, tenta localizá-lo
        if not caminho_audio:
            id_b = normalizar_id_arquivo(caminho_jams)
            candidatos = [
                os.path.join(args.dataset_dir, args.audio_type, f"{id_b}_mic.wav"),
                os.path.join(args.dataset_dir, args.audio_type, f"{id_b}.wav"),
                os.path.join(args.dataset_dir, args.audio_type, f"{id_b}_hex_cln.wav")
            ]
            for cand in candidatos:
                if os.path.exists(cand):
                    caminho_audio = cand
                    break

        if not caminho_audio or not os.path.exists(caminho_audio):
            print(f"\n[ERRO] Não foi possível localizar o áudio correspondente para '{caminho_jams}'.")
            print("Especifique o arquivo de áudio utilizando o parâmetro --audio <caminho.wav>\n")
            sys.exit(1)

        executar_avaliacao_individual(
            caminho_audio=caminho_audio,
            caminho_jams=caminho_jams,
            onset_tol=args.tolerancia_onset,
            pitch_tol=args.tolerancia_pitch,
            offset_ratio=args.tolerancia_offset,
            detalhado=args.detalhado
        )
        sys.exit(0)

    # 5. Modo Avaliação em Lote (Batch)
    if args.batch:
        filtro = None if args.style == "all" else args.style
        csv_saida = args.salvar_csv
        if not csv_saida:
            # Salva por padrão em resultados/avaliacao_guitarset.csv
            pasta_res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
            os.makedirs(pasta_res, exist_ok=True)
            csv_saida = os.path.join(pasta_res, "avaliacao_guitarset.csv")

        executar_avaliacao_lote(
            diretorio_dataset=args.dataset_dir,
            tipo_audio=args.audio_type,
            filtro_estilo=filtro,
            limite_arquivos=args.max_files,
            caminho_csv=csv_saida,
            onset_tol=args.tolerancia_onset,
            pitch_tol=args.tolerancia_pitch,
            offset_ratio=args.tolerancia_offset
        )
        sys.exit(0)

    # 6. Caso nenhum modo seja especificado: Inspeciona dataset e orienta o usuário
    valido, total_j, total_a, msg = verificar_estrutura_dataset(
        caminho_base=args.dataset_dir,
        tipo_audio=args.audio_type
    )

    print("\n" + "=" * 80)
    print(" SCRIPT DE AVALIAÇÃO DE TRANSCRIÇÃO COM GUITARSET (TCC)")
    print("=" * 80)
    print(f" Estado do Dataset em '{args.dataset_dir}':")
    print(f"   • Situação: {msg}")
    print(f"   • Anotações (.jams): {total_j}")
    print(f"   • Arquivos de Áudio: {total_a}")
    print("-" * 80)

    if not valido or total_j == 0:
        print("\n [ATENÇÃO] O dataset GuitarSet não foi encontrado na pasta padrão.")
        print(" Para visualizar instruções completas de download e organização, execute:")
        print("     python avaliar_com_guitarset.py --setup\n")
        print(" Para criar a estrutura de pastas automaticamente:")
        print("     python avaliar_com_guitarset.py --criar-pastas\n")
    else:
        print("\n [PRONTO] O dataset foi detectado com sucesso!")
        print(" Comandos recomendados para avaliação:")
        print("   1. Avaliar todo o dataset (solos melódicos):")
        print("      python avaliar_com_guitarset.py --batch --style solo")
        print("   2. Avaliar uma amostra rápida (ex: 5 arquivos):")
        print("      python avaliar_com_guitarset.py --batch --max-files 5")
        print("   3. Avaliar um arquivo específico:")
        print("      python avaliar_com_guitarset.py --jams <caminho.jams> --audio <caminho.wav>\n")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
