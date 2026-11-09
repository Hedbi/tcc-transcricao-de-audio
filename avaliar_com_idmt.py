# =============================================================================
# TCC - Transcrição de Áudio de Guitarra para Tablatura
# Módulo de Validação e Benchmark: IDMT-SMT-GUITAR (Notas Isoladas Monofônicas)
# =============================================================================
import os
import glob
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import pipeline_transcritor as pt

def avaliar_idmt_monofonico(
    pasta_raiz_dataset: str,
    max_amostras: int = 50,
    salvar_csv: str = None
):
    pasta_audio = os.path.join(pasta_raiz_dataset, "dataset1", "Fender Strat Clean Neck SC", "audio")
    pasta_annot = os.path.join(pasta_raiz_dataset, "dataset1", "Fender Strat Clean Neck SC", "annotation")

    if not os.path.exists(pasta_audio):
        print(f"[ERRO] Diretório de áudio não encontrado: {pasta_audio}")
        return

    arquivos_audio = sorted(glob.glob(os.path.join(pasta_audio, "*.wav")))
    if max_amostras:
        arquivos_audio = arquivos_audio[:max_amostras]

    print("\n" + "="*80)
    print(f"AVALIAÇÃO DO PIPELINE NO BENCHMARK IDMT-SMT-GUITAR (N = {len(arquivos_audio)})")
    print("="*80)

    registros = []
    total_acertos_pitch = 0
    total_avaliados = 0

    for a in arquivos_audio:
        nome_base = os.path.splitext(os.path.basename(a))[0]
        xml_path = os.path.join(pasta_annot, nome_base + ".xml")

        if not os.path.exists(xml_path):
            continue

        tree = ET.parse(xml_path)
        ev = tree.getroot().find(".//event")
        gt_pitch = int(ev.find("pitch").text)
        gt_onset = float(ev.find("onsetSec").text)
        gt_offset = float(ev.find("offsetSec").text)
        gt_string = int(ev.find("stringNumber").text)
        gt_fret = int(ev.find("fretNumber").text)

        notas = pt.transcrever_audio(a)
        total_avaliados += 1

        # Para nota isolada, seleciona o evento com maior duração (sustentação estável)
        if notas:
            nota_principal = max(notas, key=lambda n: n["duration"])
            pred_pitch = nota_principal["midi_number"]
            pred_onset = nota_principal["onset_time"]
            pred_dur = nota_principal["duration"]
            pred_nome = nota_principal["note_name"]
            acertou = (pred_pitch == gt_pitch)
        else:
            nota_principal = None
            pred_pitch = None
            pred_onset = None
            pred_dur = 0.0
            pred_nome = "None"
            acertou = False

        if acertou:
            total_acertos_pitch += 1

        registros.append({
            "arquivo": nome_base + ".wav",
            "gt_pitch": gt_pitch,
            "gt_string": gt_string,
            "gt_fret": gt_fret,
            "gt_onset": gt_onset,
            "gt_offset": gt_offset,
            "pred_nome": pred_nome,
            "pred_pitch": pred_pitch,
            "pred_onset": pred_onset,
            "pred_dur": pred_dur,
            "acertou_pitch": acertou
        })

        status = "ACERTOU" if acertou else "ERROU"
        print(f"[{status:7s}] {nome_base}.wav | Gabarito MIDI: {gt_pitch:2d} (Corda {gt_string}, Casa {gt_fret:2d}) | Predito: {pred_nome} (MIDI {pred_pitch})")

    acuracia = (total_acertos_pitch / total_avaliados * 100) if total_avaliados > 0 else 0

    print("\n" + "="*80)
    print("RESUMO QUANTITATIVO DO BENCHMARK:")
    print(f"  Total de Amostras Avaliadas: {total_avaliados}")
    print(f"  Acertos Exatos de Pitch:     {total_acertos_pitch}")
    print(f"  Acurácia de Pitch (%):       {acuracia:.2f}%")
    print("="*80 + "\n")

    if salvar_csv:
        os.makedirs(os.path.dirname(os.path.abspath(salvar_csv)), exist_ok=True)
        df = pd.DataFrame(registros)
        df.to_csv(salvar_csv, index=False, encoding="utf-8")
        print(f"Relatório exportado para CSV: {salvar_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliar pipeline com dataset IDMT-SMT-GUITAR.")
    parser.add_argument("--dataset-dir", default=r"dataset/IDMT-SMT-GUITAR_V2", help="Pasta raiz do dataset.")
    parser.add_argument("--max", type=int, default=20, help="Número máximo de arquivos para testar.")
    parser.add_argument("--salvar-csv", default=r"resultados/benchmark_idmt_monofonico.csv", help="Caminho do CSV de saída.")
    args = parser.parse_args()

    avaliar_idmt_monofonico(args.dataset_dir, max_amostras=args.max, salvar_csv=args.salvar_csv)