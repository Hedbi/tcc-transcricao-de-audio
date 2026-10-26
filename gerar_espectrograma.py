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
# 'ndimage' (N-Dimensional Image) oferece a operação de abertura morfológica
# usada apenas na comparação exploratória da Máscara Binária.
# Referência: Gonzalez & Woods, "Processamento Digital de Imagens", Cap. 9.
# -----------------------------------------------------------------------------
from scipy.ndimage import binary_opening, label, find_objects

# BIBLIOTECAS DE ÁUDIO E VISUALIZAÇÃO
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches


# =============================================================================
# FUNÇÃO PRINCIPAL: Carrega o áudio, gera o CQT e aplica a Máscara Binária
# =============================================================================
def gerar_imagem_do_audio(
    caminho_audio,
    caminho_imagem_saida,
    limiar_db=-40,
    aplicar_morfologia=False
):
    """
    Pipeline completo de pré-processamento espectral para guitarra.

    Parâmetros:
        caminho_audio (str):         Caminho do arquivo WAV de entrada.
        caminho_imagem_saida (str):  Caminho onde a imagem comparativa será salva.
        limiar_db (float):           Limiar de corte em decibéis para a Máscara Binária.
                                     Padrão: -40 dB. (Ver justificativa abaixo.)
        aplicar_morfologia (bool):   Ativa a limpeza morfológica apenas para
                                     comparação exploratória. Desativada por
                                     padrão para não alterar ataques.
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
    # -40 dB é uma configuração exploratória relativa ao pico, não uma
    # calibração validada. A escolha definitiva deve ser comparada por métricas
    # em um conjunto anotado; a transcrição principal não depende desta máscara.
    mascara_binaria = (espectrograma_db > limiar_db).astype(np.uint8)

    # =========================================================================
    # ETAPA 5: POLIMENTO DA MÁSCARA (Morfologia Matemática)
    # =========================================================================
    # Apesar do limiar já filtrar boa parte do ruído, a máscara bruta ainda
    # apresenta dois problemas acustico-visuais criticos:
    # A) O transiente de palhetada (ataque) gera uma barra vertical de banda larga
    #    que toca todos os harmonicos, atuando como uma "ponte" indesejada.
    # B) O vibrato natural e decaimento da nota causam pequenos buracos horizontais.
    # A Morfologia Matematica resolve ambos com operadores direcionais.

    # =========================================================================
    # OPERACAO 5.1: ABERTURA HORIZONTAL (Desacoplamento de Transientes)
    # =========================================================================
    # PARÂMETRO CRITICO: structure=np.ones((1, 5)) -> Bloco 1 linha x 5 colunas
    # Justificativa: Ao tocar a corda, o atrito da palheta cria um estalo
    # percussivo vertical que dura apenas 1 a 2 frames (~23 a 46ms). Se
    # aplicassemos fechamento direto, essa linha vertical fundiria a fundamental
    # e todos os harmonicos em um unico bloco gigante conectado (formato de pente).
    # Uma abertura com elemento estritamente horizontal (1x5, duracao >= 115ms)
    # apaga os transientes verticais curtos e preserva apenas as notas que tem
    # sustentacao acustica horizontal no tempo.
    # Referencia: Gonzalez & Woods, "Processamento Digital de Imagens", Cap. 9.
    if aplicar_morfologia:
        # A abertura 1x5 remove estruturas com menos de cinco frames contínuos
        # na direção temporal. Ela não preserva manchas curtas de 1 ou 2 pixels.
        mascara_apos_abertura = binary_opening(
            mascara_binaria, structure=np.ones((1, 5))
        )
        # Não há fechamento por padrão: qualquer kernel horizontal pode unir
        # ataques consecutivos e apagar a separação temporal dos eventos.
        mascara_limpa = mascara_apos_abertura
    else:
        mascara_limpa = mascara_binaria

    # O fechamento horizontal foi deliberadamente removido. Mesmo uma lacuna
    # de poucos frames pode separar dois ataques legítimos; preenchê-la faria
    # a imagem perder exatamente a informação temporal que a Etapa 2 precisa.

    # =========================================================================
    # ETAPA 6: APLICACAO DA MASCARA NO ESPECTROGRAMA ORIGINAL
    # =========================================================================
    # A mascara age como um estencil perfeito: onde mascara_limpa == 1,
    # preservamos a amplitude em dB exata da nota. Onde for 0, forca para -80 dB.
    espectrograma_limpo = np.where(mascara_limpa, espectrograma_db, -80)

    # =========================================================================
    # ETAPA 7: DETECCAO DE ONSETS, OFFSETS E PITCH TRACKING (ETAPA 2 DO TCC)
    # =========================================================================
    # Abordagem de Visao Computacional baseada em Componentes Conexos.
    # Cada "ilha" de pixels brancos na mascara representa um evento acustico
    # (seja a nota fundamental ou um de seus sobretons harmonicos).
    rotulado, num_componentes = label(mascara_limpa)
    fatias_objetos = find_objects(rotulado)

    componentes_validos = []
    for idx, fatia in enumerate(fatias_objetos):
        if fatia is None:
            continue
        
        # Filtro de Area: Elimina ruidos residuais ou estalos mecanicos
        # que tenham duracao e espessura inferiores a 40 pixels.
        area_px = np.sum(rotulado[fatia] == (idx + 1))
        if area_px < 40:
            continue

        fatia_bin, fatia_tempo = fatia
        t_onset = fatia_tempo.start * 512 / sr
        t_offset = fatia_tempo.stop * 512 / sr
        duracao = t_offset - t_onset

        # Extrai o bin de maior intensidade energetica dentro deste componente
        mascara_comp = (rotulado[fatia] == (idx + 1))
        dados_comp = np.where(mascara_comp, espectrograma_db[fatia], -120)
        pico_relativo = np.unravel_index(np.argmax(dados_comp), dados_comp.shape)[0]
        bin_pico = fatia_bin.start + pico_relativo

        # Conversao do bin CQT para nome da nota musical e frequencia Hz
        # unicode=False garante compatibilidade de caracteres (# em vez de ♯) no Windows
        nome_nota = librosa.midi_to_note(librosa.note_to_midi('E2') + bin_pico, unicode=False)
        freq_hz = librosa.cqt_frequencies(n_bins=84, fmin=nota_mi_grave)[bin_pico]

        componentes_validos.append({
            'id': idx + 1,
            'bin_min': fatia_bin.start,
            'bin_max': fatia_bin.stop,
            'frame_min': fatia_tempo.start,
            'frame_max': fatia_tempo.stop,
            'bin_pico': bin_pico,
            'nome_nota': nome_nota,
            'freq_hz': freq_hz,
            't_onset': t_onset,
            't_offset': t_offset,
            'duracao': duracao,
            'area_px': int(area_px)
        })

    # Ordena os componentes cronologicamente pelo tempo de onset
    componentes_validos.sort(key=lambda c: c['t_onset'])

    # -------------------------------------------------------------------------
    # AGRUPAMENTO HARMONICO (Resolucao do Problema da Serie Harmonica)
    # Justificativa Acustica: Quando uma corda vibra, a Fundamental (f0) e seus
    # sobretons (2f0, 3f0...) nascem exatamente juntos no tempo da palhetada.
    # Agrupamos componentes que iniciam em uma mesma janela temporal (<= 0.35s).
    # O componente com a MENOR frequencia e classificado como a Fundamental (f0).
    # -------------------------------------------------------------------------
    eventos_musicais = []
    componentes_usados = set()

    for i, comp in enumerate(componentes_validos):
        if i in componentes_usados:
            continue
        grupo = [comp]
        componentes_usados.add(i)

        for j, outro in enumerate(componentes_validos):
            if j in componentes_usados:
                continue
            # Verifica concorrencia temporal (ataque similar e sobreposicao no tempo)
            if abs(outro['t_onset'] - comp['t_onset']) <= 0.35:
                sobreposicao = min(outro['t_offset'], comp['t_offset']) - max(outro['t_onset'], comp['t_onset'])
                if sobreposicao > 0:
                    grupo.append(outro)
                    componentes_usados.add(j)

        # Ordena o grupo da frequencia mais grave para a mais aguda
        grupo.sort(key=lambda c: c['bin_pico'])
        fundamental = grupo[0]
        harmonicos = grupo[1:]

        eventos_musicais.append({
            'fundamental': fundamental,
            'harmonicos': harmonicos
        })

    # =========================================================================
    # ETAPA 8: GERAÇÃO DA IMAGEM COMPARATIVA (4 Painéis com Bounding Boxes)
    # =========================================================================
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
    ax1.set_title('1. Espectrograma Original CQT (Sem Filtro)', fontsize=11, fontweight='bold')
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
    titulo_mascara = (
        '3. Mascara com Abertura Exploratoria (1x5)'
        if aplicar_morfologia else
        '3. Mascara Binaria sem Morfologia (fluxo nao destrutivo)'
    )
    ax3.set_title(titulo_mascara, fontsize=11, fontweight='bold')
    ax3.set_xlabel('Frames (unidades de tempo)')
    ax3.set_ylabel('Bins CQT (unidades de frequencia)')

    # --- Painel 4 (Inferior Direito): Espectrograma Final + Bounding Boxes das Notas ---
    ax4 = fig.add_subplot(gs[1, 1])
    img4 = librosa.display.specshow(
        espectrograma_limpo, sr=sr, hop_length=512,
        x_axis='time', y_axis='cqt_note', fmin=nota_mi_grave,
        cmap='magma', ax=ax4
    )
    ax4.set_title('4. Espectrograma Final + Deteccao de Notas (Etapa 2)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Tempo (segundos)')
    ax4.set_ylabel('Notas Musicais')
    fig.colorbar(img4, ax=ax4, format='%+2.0f dB')

    # Desenha Bounding Boxes ao redor das notas fundamentais detectadas no Painel 4
    for ev in eventos_musicais:
        f = ev['fundamental']
        t_ini = f['t_onset']
        t_dur = f['duracao']
        # Conversao aproximada do bin para a coordenada vertical de exibicao no specshow
        # No specshow com y_axis='cqt_note', a coordenada y interna mapeia as frequencias em Hz
        f_min_box = librosa.cqt_frequencies(n_bins=84, fmin=nota_mi_grave)[f['bin_min']]
        f_max_box = librosa.cqt_frequencies(n_bins=84, fmin=nota_mi_grave)[min(f['bin_max'], 83)]
        h_box = f_max_box - f_min_box

        retangulo = patches.Rectangle(
            (t_ini, f_min_box), t_dur, h_box,
            linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'
        )
        ax4.add_patch(retangulo)
        ax4.text(
            t_ini, f_max_box * 1.05, f"{f['nome_nota']} ({f['freq_hz']:.0f}Hz)",
            color='cyan', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7, edgecolor='cyan')
        )

    fig.suptitle(
        'Pipeline Integrada: CQT -> Mascara Morfologica -> Bounding Boxes de Notas (TCC)',
        fontsize=13, fontweight='bold', y=0.98
    )

    plt.savefig(caminho_imagem_saida, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[SUCESSO] Imagem comparativa gerada em: {caminho_imagem_saida}")
    print(f"  > Componentes detectados na mascara: {len(componentes_validos)}")
    print(f"  > Notas fundamentais identificadas:   {len(eventos_musicais)}")
    print("\n" + "=" * 70)
    print("TABELA DE NOTAS EXTRAIDAS (ETAPA 2 - PITCH & ONSET TRACKING)")
    print("=" * 70)
    for idx, ev in enumerate(eventos_musicais, 1):
        f = ev['fundamental']
        harms = [f"{h['nome_nota']} ({h['freq_hz']:.1f}Hz)" for h in ev['harmonicos']]
        harms_str = ", ".join(harms) if harms else "Nenhum"
        print(f"Nota {idx}: {f['nome_nota']} ({f['freq_hz']:.1f} Hz)")
        print(f"   -> Onset (Ataque):  {f['t_onset']:.2f} s")
        print(f"   -> Offset (Termino):{f['t_offset']:.2f} s")
        print(f"   -> Duracao Real:    {f['duracao']:.2f} s")
        print(f"   -> Harmonicos Filtrados: [{harms_str}]")
        print("-" * 70)

    return eventos_musicais


# =============================================================================
# BLOCO DE EXECUÇÃO
# =============================================================================
if __name__ == '__main__':
    # Executa a pipeline sobre o audio selecionado e salva o resultado sequencialmente.
    diretorio_base = os.path.dirname(os.path.abspath(__file__))
    
    # Pode alternar aqui entre 'som2.wav', 'som4.mp3', etc.
    audio_entrada = os.path.join(diretorio_base, 'audios', 'som2.wav')

    pasta_resultados = os.path.join(diretorio_base, 'resultados')
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
    caminho_saida = os.path.join(pasta_resultados, f'resultado_{proximo_numero}.png')

    eventos_detectados = gerar_imagem_do_audio(audio_entrada, caminho_saida)
