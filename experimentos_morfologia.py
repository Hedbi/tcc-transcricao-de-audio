"""Diagnóstico reprodutível dos parâmetros da máscara CQT.

Este script não afirma que um parâmetro é melhor apenas por produzir mais
componentes. Ele registra medidas descritivas para orientar a análise
exploratória; a escolha final deve ser validada com o GuitarSet e
``avaliar_com_guitarset.py``.
"""

import argparse
import csv
import os

import librosa
import numpy as np
from scipy.ndimage import binary_opening, label


def medir_configuracao(
    caminho_audio: str,
    limiar_db: float,
    abertura_frames: int
) -> dict[str, float | int]:
    """Calcula medidas da máscara sem fechamento morfológico."""
    y, sr = librosa.load(caminho_audio, sr=22050, mono=True)
    cqt = librosa.cqt(
        y, sr=sr, fmin=librosa.note_to_hz("E2"), n_bins=84, hop_length=512
    )
    espectrograma_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    mascara = espectrograma_db > limiar_db

    if abertura_frames > 1:
        mascara = binary_opening(
            mascara, structure=np.ones((1, abertura_frames), dtype=bool)
        )

    _, componentes = label(mascara)
    return {
        "limiar_db": limiar_db,
        "abertura_frames": abertura_frames,
        "frames_ativos": int(np.count_nonzero(mascara)),
        "proporcao_ativa": float(np.mean(mascara)),
        "componentes_conexos": int(componentes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera uma tabela descritiva dos parâmetros da máscara CQT."
    )
    parser.add_argument("audio", help="Arquivo de áudio a analisar.")
    parser.add_argument(
        "--saida",
        default="resultados/experimentos_morfologia.csv",
        help="CSV de saída.",
    )
    parser.add_argument(
        "--limiares",
        type=float,
        nargs="+",
        default=[-30.0, -40.0, -50.0],
        help="Limiares em dB relativos ao pico.",
    )
    parser.add_argument(
        "--aberturas",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Larguras temporais da abertura; 1 desativa a operação.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {args.audio}")

    resultados = [
        medir_configuracao(args.audio, limiar, abertura)
        for limiar in args.limiares
        for abertura in args.aberturas
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.saida)), exist_ok=True)
    with open(args.saida, "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)

    print(f"[SUCESSO] Diagnóstico salvo em: {os.path.abspath(args.saida)}")
    print("Use avaliar_com_guitarset.py para medir F1 de onset e pitch.")


if __name__ == "__main__":
    main()
