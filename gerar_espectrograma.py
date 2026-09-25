# =============================================================================
# TCC - Transcrição de Áudio de Guitarra para Tablatura
# Etapa 1: Geração, Limpeza e Pré-processamento do Espectrograma CQT
# =============================================================================
# BIBLIOTECAS DO SISTEMA
import os
import glob

# BIBLIOTECAS CIENTÍFICAS
import numpy as np

# -----------------------------------------------------------------------------
# JUSTIFICATIVA TEÓRICA (para a escrita da monografia):
# O SciPy (Scientific Python) é a biblioteca de computação científica padrão
# para o processamento de sinais discretos e morfologia matemática. O módulo
# 'ndimage' (N-Dimensional Image) oferece as operações de abertura e fechamento
# morfológico que serão aplicadas sobre a Máscara Binária.
# Referência: Gonzalez & Woods, "Processamento Digital de Imagens", Cap. 9.
# -----------------------------------------------------------------------------
from scipy.ndimage import binary_closing, binary_opening

# BIBLIOTECAS DE ÁUDIO E VISUALIZAÇÃO
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# FUNÇÃO PRINCIPAL: Carrega o áudio, gera o CQT e aplica a Máscara Binária
# =============================================================================
def gerar_imagem_do_audio(caminho_audio, caminho_imagem_saida, limiar_db=-40):
    """
    Pipeline completo de pré-processamento espectral para guitarra.

    Parâmetros:
        caminho_audio (str):         Caminho do arquivo WAV de entrada.
        caminho_imagem_saida (str):  Caminho onde a imagem comparativa será salva.
        limiar_db (float):           Limiar de corte em decibéis para a Máscara Binária.
                                     Padrão: -40 dB. (Ver justificativa abaixo.)
    """

    # =========================================================================
    # ETAPA 1: CARREGAMENTO DO ÁUDIO
    # =========================================================================
    # PARÂMETRO CRÍTICO: sr=22050 Hz
    # Justificativa: O Teorema de Nyquist-Shannon estabelece que a taxa de
    # amostragem (sr) deve ser pelo menos o dobro da frequência mais alta que
    # se deseja capturar (fmax = sr / 2). Para a guitarra, os harmônicos
    # musicalmente relevantes raramente ultrapassam 8.000 Hz. Usar 22.050 Hz
    # garante uma margem segura de captura até 11.025 Hz, e ao mesmo tempo
    # reduz o arquivo de dados à metade do tamanho em relação a um CD de
    # áudio padrão (44.100 Hz), acelerando toda a pipeline de processamento.
    # Referência: Oppenheim & Schafer, "Discrete-Time Signal Processing".
    y, sr = librosa.load(caminho_audio, sr=22050)

    # =========================================================================
    # ETAPA 2: TRANSFORMAÇÃO CQT (Constant-Q Transform)
    # =========================================================================
    # PARÂMETRO CRÍTICO: fmin = nota 'E2' (~82,41 Hz)
    # Justificativa: Na afinação padrão da guitarra de 6 cordas (E-A-D-G-B-e),
    # a nota mais grave possível é o Mi da 6ª corda solta (E2 ≈ 82,41 Hz).
    # Definir fmin nesse valor descarta toda a energia abaixo de 82 Hz, que
    # na prática é apenas ruído mecânico do instrumento ou vibração de baixa
    # frequência, eliminando uma fonte de poluição espectral logo na origem.
    nota_mi_grave = librosa.note_to_hz('E2')

    # PARÂMETRO CRÍTICO: n_bins=84
    # Justificativa: A CQT usa, por padrão, 12 bins por oitava (um bin para
    # cada um dos 12 semitons da escala cromática temperada). Com 84 bins,
    # temos 84 / 12 = 7 oitavas completas. Isso cobre de E2 (~82 Hz) até
    # aproximadamente E9 (~10.500 Hz), englobando todas as notas fundamentais
    # tocáveis na guitarra e os principais harmônicos de cada uma.
    # Diferente da FFT (resolução linear em Hz), a CQT mantém a razão
    # Q = f / delta_f constante, o que a torna naturalmente alinhada à escala
    # musical logarítmica. Essa é a sua grande vantagem sobre a STFT clássica.
    # Referência: Brown, J.C. (1991). "Calculation of a Constant Q Spectral
    # Transform", Journal of the Acoustical Society of America.
    #
    # PARÂMETRO CRÍTICO: hop_length=512
    # Justificativa: O hop_length define o passo em amostras entre cada
    # 'janela' de análise da CQT (ou seja, de quantas em quantas amostras o
    # algoritmo "olha" para o áudio novamente).
    # Com sr=22050 e hop_length=512, a resolução temporal da imagem é de:
    # 512 / 22050 ≈ 0,023 segundos por coluna (≈ 23 milissegundos por frame).
    # Para guitarra, isso é suficiente: uma nota muito rápida em 180 BPM dura
    # cerca de 83ms (uma semicolcheia), o que equivale a ~3 colunas no
    # espectrograma, garantindo que nenhuma nota seja "perdida" entre dois
    # frames. Valores menores (ex: 256) dariam mais resolução temporal, mas
    # dobrariam o tamanho da matriz e o tempo de processamento sem ganho
    # relevante para as notas da guitarra.
    cqt_bruto = librosa.cqt(y, sr=sr, fmin=nota_mi_grave, n_bins=84, hop_length=512)

    # =========================================================================
    # ETAPA 3: CONVERSÃO PARA DECIBÉIS
    # =========================================================================
    # np.abs() extrai a magnitude do número complexo (descarta a fase).
    # ref=np.max normaliza o espectrograma de forma que o ponto de maior
    # energia do áudio inteiro receba o valor de 0 dB. Todos os outros
    # valores serão negativos (ex: -10 dB, -40 dB).
    # Justificativa: A escala logarítmica em decibéis reflete a percepção
    # auditiva humana (Lei de Weber-Fechner) e "abre" a escala visual para
    # revelar tanto os picos fortes quanto os harmônicos sutis na mesma imagem.
    espectrograma_db = librosa.amplitude_to_db(np.abs(cqt_bruto), ref=np.max)

    # =========================================================================
    # ETAPA 4: CRIAÇÃO DA MÁSCARA BINÁRIA (Binary Time-Frequency Mask)
    # =========================================================================
    # Uma Máscara Binária é uma matriz de MESMO TAMANHO que o espectrograma,
    # mas contendo APENAS zeros (0) ou uns (1).
    # - 1 (Verdadeiro)  = Essa região de tempo-frequência contém sinal musical.
    # - 0 (Falso)       = Essa região é ruído, eco ou silêncio.
    #
    # A comparação (espectrograma_db > limiar_db) varre cada célula da matriz
    # e retorna True onde a intensidade for maior que o limiar, e False onde
    # não for. O .astype(np.uint8) converte True→1 e False→0.
    #
    # PARÂMETRO CRÍTICO: limiar_db = -40 dB
    # Justificativa: O limiar de -40 dB (relativo ao pico de 0 dB) foi
    # calibrado empiricamente para a guitarra acústica/elétrica com base
    # no seguinte raciocínio: os primeiros 2 ou 3 harmônicos de uma nota de
    # guitarra tipicamente ficam entre -5 dB e -25 dB. O eco de sala e ruído
    # de cauda ficam abaixo de -45 dB a -80 dB. O limiar de -40 dB posiciona
    # o corte no "vale" entre esses dois grupos, capturando os harmônicos
    # principais sem capturar o eco. Isso é equivalente ao conceito de Noise
    # Gate (portão de ruído) na indústria de áudio profissional.
    # Referência: Zölzer, U. "DAFX: Digital Audio Effects", Cap. 5 (Dynamics).
    mascara_binaria = (espectrograma_db > limiar_db).astype(np.uint8)

    # =========================================================================
    # ETAPA 5: POLIMENTO DA MÁSCARA (Morfologia Matemática)
    # =========================================================================
    # Apesar do limiar já filtrar boa parte do ruído, a máscara bruta ainda
    # apresenta dois problemas visuais/computacionais:
    # A) Notas longas têm pequenos "buracos" no meio (descontinuidades).
    # B) Existem "ilhas" de pixels isolados que são estalos ou ruído elétrico.
    # A Morfologia Matemática resolve ambos com dois operadores clássicos.

    # =========================================================================
    # OPERAÇÃO 5.1: FECHAMENTO MORFOLÓGICO (Binary Closing)
    # =========================================================================
    # O FECHAMENTO é a composição de DILATAÇÃO seguida de EROSÃO.
    # Na prática, ele PREENCHE buracos e rachaduras nas manchas brancas (notas)
    # SEM aumentar o tamanho geral da nota.
    #
    # PARÂMETRO CRÍTICO: structure=np.ones((1, 20))  →  Bloco 1 linha × 20 colunas
    # Justificativa: O elemento estruturante define "qual formato de buraco"
    # o algoritmo vai tentar preencher. Usamos um bloco HORIZONTAL (1×20)
    # porque as notas da guitarra formam linhas horizontais no espectrograma
    # (pitch fixo ao longo do tempo). Um buraco horizontal de até 20 colunas
    # equivale a uma falha de até 20 × 23ms ≈ 0,46 segundos de duração.
    # Esse valor foi escolhido para "costurar" falhas causadas pelo tremido
    # natural de uma nota sustentada (vibrato sutil que cai abaixo do limiar
    # por frações de segundo), mas NÃO é grande o suficiente para conectar
    # duas notas distintas separadas por uma pausa real (que tipicamente dura
    # mais de 0,5s em uma execução normal de guitarra).
    # Referência: Gonzalez & Woods, "Processamento Digital de Imagens", Cap. 9.
    mascara_apos_fechamento = binary_closing(mascara_binaria, structure=np.ones((1, 20)))

    # =========================================================================
    # OPERAÇÃO 5.2: ABERTURA MORFOLÓGICA (Binary Opening)
    # =========================================================================
    # A ABERTURA é a composição de EROSÃO seguida de DILATAÇÃO.
    # Na prática, ela APAGA manchas brancas pequenas e isoladas (ruídos)
    # SEM encolher as manchas grandes (as notas reais).
    #
    # PARÂMETRO CRÍTICO: structure=np.ones((3, 3))  →  Bloco 3×3
    # Justificativa: Qualquer "ilha" de pixels brancos com menos de 3 colunas
    # de largura (≈ 69ms) ou menos de 3 linhas de altura (≈ 3 semitons de
    # espessura) provavelmente não é uma nota musical real — é um estalo de
    # palheta, interferência elétrica ou artefato da transformada. O bloco
    # 3×3 foi calibrado para ser pequeno o suficiente para NÃO apagar os
    # harmônicos superiores mais finos (que têm 1 ou 2 bins de espessura),
    # mas grande o suficiente para eliminar o chiado pontual.
    mascara_limpa = binary_opening(mascara_apos_fechamento, structure=np.ones((3, 3)))

    # =========================================================================
    # ETAPA 6: APLICAÇÃO DA MÁSCARA NO ESPECTROGRAMA ORIGINAL
    # =========================================================================
    # PONTO-CHAVE: A máscara NÃO modifica os valores de decibéis das notas.
    # Ela age como um estêncil: onde é 1, o valor original permanece intacto.
    # Onde é 0, a intensidade é forçada para -80 dB (silêncio visual).
    # Isso garante que a precisão musical do espectrograma seja preservada.
    #
    # A operação np.where funciona como um if-else vetorizado:
    # - Onde mascara_limpa == 1 → mantém o valor de espectrograma_db
    # - Onde mascara_limpa == 0 → substitui por -80 dB
    espectrograma_limpo = np.where(mascara_limpa, espectrograma_db, -80)

    # =========================================================================
    # ETAPA 7: GERAÇÃO DA IMAGEM COMPARATIVA (4 Painéis)
    # =========================================================================
    # Para fins de análise e apresentação no TCC, geramos um único arquivo
    # de imagem com quatro painéis que mostram cada etapa da pipeline:
    # Painel 1: Espectrograma original (sem filtro)
    # Painel 2: Máscara binária bruta (preto e branco, logo após o limiar)
    # Painel 3: Máscara polida (após fechamento e abertura morfológica)
    # Painel 4: Espectrograma final limpo (máscara aplicada sobre os dBs)
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # --- Painel 1 (Superior Esquerdo): Espectrograma Original ---
    espectrograma_original = librosa.amplitude_to_db(np.abs(cqt_bruto), ref=np.max)
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = librosa.display.specshow(
        espectrograma_original, sr=sr, hop_length=512,
        x_axis='time', y_axis='cqt_note', fmin=nota_mi_grave,
        cmap='magma', ax=ax1
    )
    ax1.set_title('1. Espectrograma Original (Sem Filtro)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Tempo (segundos)')
    ax1.set_ylabel('Notas Musicais')
    fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

    # --- Painel 2 (Superior Direito): Máscara Binária Bruta ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(
        mascara_binaria,
        origin='lower', aspect='auto',
        cmap='gray', interpolation='nearest'
    )
    ax2.set_title('2. Mascara Binaria Bruta (Limiar -40 dB)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Frames (unidades de tempo)')
    ax2.set_ylabel('Bins CQT (unidades de frequencia)')

    # --- Painel 3 (Inferior Esquerdo): Máscara Polida pela Morfologia ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(
        mascara_limpa.astype(np.uint8),
        origin='lower', aspect='auto',
        cmap='gray', interpolation='nearest'
    )
    ax3.set_title('3. Mascara Polida (Fechamento + Abertura Morfologica)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Frames (unidades de tempo)')
    ax3.set_ylabel('Bins CQT (unidades de frequencia)')

    # --- Painel 4 (Inferior Direito): Espectrograma Final Limpo ---
    ax4 = fig.add_subplot(gs[1, 1])
    img4 = librosa.display.specshow(
        espectrograma_limpo, sr=sr, hop_length=512,
        x_axis='time', y_axis='cqt_note', fmin=nota_mi_grave,
        cmap='magma', ax=ax4
    )
    ax4.set_title('4. Espectrograma Final (Mascara Aplicada)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Tempo (segundos)')
    ax4.set_ylabel('Notas Musicais')
    fig.colorbar(img4, ax=ax4, format='%+2.0f dB')

    fig.suptitle(
        'Pipeline de Limpeza Espectral: CQT + Mascara Binaria + Morfologia Matematica',
        fontsize=13, fontweight='bold', y=0.98
    )

    plt.savefig(caminho_imagem_saida, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Imagem comparativa (4 paineis) gerada com sucesso em: {caminho_imagem_saida}")
    print(f"  > Total de pixels ativos na mascara bruta:  {mascara_binaria.sum()}")
    print(f"  > Total de pixels ativos na mascara limpa:  {mascara_limpa.sum()}")
    print(f"  > Pixels removidos pela morfologia:         {mascara_binaria.sum() - mascara_limpa.sum()}")


# =============================================================================
# BLOCO DE EXECUÇÃO
# =============================================================================
# Usa o caminho absoluto relativo ao local deste script, evitando erros
# causados por executar o arquivo de um diretório diferente no terminal.
diretorio_base = os.path.dirname(os.path.abspath(__file__))
audio_entrada = os.path.join(diretorio_base, 'audios', 'som4.mp3')

# Cria a pasta de resultados se não existir
pasta_resultados = os.path.join(diretorio_base, 'resultados')
os.makedirs(pasta_resultados, exist_ok=True)

# Descobre o próximo número de resultado para não sobrescrever os anteriores
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
caminho_saida = os.path.join(pasta_resultados, f'resultado_{proximo_numero}.png')

gerar_imagem_do_audio(audio_entrada, caminho_saida)
