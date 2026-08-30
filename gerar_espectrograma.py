import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def gerar_imagem_do_audio(caminho_audio, caminho_imagem_saida):
    # 1. Carregamento do áudio
    # sr=22050 é a taxa de amostragem padrão e suficiente para guitarra
    y, sr = librosa.load(caminho_audio, sr=22050)
    
    # 2. Transformação CQT (Áudio para Frequências Musicais)
    # fmin foca na nota mais grave da guitarra (Mi na 6ª corda)
    nota_mi_grave = librosa.note_to_hz('E2')
    cqt_bruto = librosa.cqt(y, sr=sr, fmin=nota_mi_grave, n_bins=84)
    
    # 3. Conversão de Amplitude para Decibéis
    # O ouvido humano percebe o som em escala logarítmica (dB)
    espectrograma_db = librosa.amplitude_to_db(np.abs(cqt_bruto), ref=np.max)
    
    # 4. Geração e Formatação da Imagem
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        espectrograma_db, 
        sr=sr, 
        x_axis='time', 
        y_axis='cqt_note', 
        fmin=nota_mi_grave,
        cmap='magma' # Paleta de cores que destaca bem as notas
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma CQT - Notas da Guitarra')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Notas Musicais')
    
    # Salva a imagem no disco e fecha a figura da memória
    plt.tight_layout()
    plt.savefig(caminho_imagem_saida, dpi=300)
    plt.close()
    
    print(f"Imagem gerada com sucesso em: {caminho_imagem_saida}")

import os
import glob

# Execução do script
audio_entrada = 'audios/som3.wav'

# Cria a pasta de resultados se não existir
pasta_resultados = 'resultados'
os.makedirs(pasta_resultados, exist_ok=True)

# Descobre o próximo número de resultado
arquivos_existentes = glob.glob(os.path.join(pasta_resultados, 'resultado_*.png'))
numeros = []
for arq in arquivos_existentes:
    try:
        # Extrai o número do nome do arquivo (ex: resultado_1.png -> 1)
        nome_base = os.path.basename(arq)
        num = int(nome_base.replace('resultado_', '').replace('.png', ''))
        numeros.append(num)
    except ValueError:
        pass

proximo_numero = 1 if not numeros else max(numeros) + 1
caminho_saida = os.path.join(pasta_resultados, f'resultado_{proximo_numero}.png')

gerar_imagem_do_audio(audio_entrada, caminho_saida)