# =============================================================================
# TCC - Transcrição Automática de Áudio de Guitarra para Tablatura
# Módulo: Mapeador de Braço da Guitarra e Gerador de Tablatura (Etapa 3)
# =============================================================================
"""
Módulo de Resolução de Ambiguidade de Braço (Fretboard Ambiguity)
e Geração de Tablatura ASCII para Transcrição de Guitarra.

Fundamentação Teórica para o TCC:
---------------------------------
Diferente do teclado/piano, onde cada tecla física corresponde a uma única nota
MIDI de frequência exclusiva, na guitarra ocorre a "Fretboard Ambiguity":
a mesma nota exata (mesmo pitch e mesma oitava) pode ser executada em múltiplas
combinações de (corda, casa). Por exemplo, a nota E4 (MIDI 64) existe em até 5
posições diferentes (1ª corda solta, 2ª corda casa 5, 3ª corda casa 9, etc.).

Este módulo implementa a heurística do Menor Esforço Biomecânico (Transição de Menor Custo):
  1. Mapeamento físico completo do braço com 6 cordas (afinação padrão EADGBE)
     e até 22 trastes.
  2. Para a primeira nota, privilegia a posição aberta/primeira posição (casas 0 a 4).
  3. Para notas sucessivas, seleciona a posição física que minimiza o deslocamento
     longitudinal da mão (distância euclidiana em trastes entre a posição anterior e a nova).
  4. Renderizador textual formatado com as 6 cordas da tablatura clássica (ASCII Tab).
"""

from typing import List, Dict, Tuple, Optional

# Definição das cordas na afinação padrão (E-A-D-G-B-E)
# Índice 0 = 6ª corda (E2 grave, MIDI 40)
# Índice 5 = 1ª corda (E4 agudo, MIDI 64)
# Notação tradicional: 6 = E2, 5 = A2, 4 = D3, 3 = G3, 2 = B3, 1 = E4
NOMES_CORDAS = ['E', 'A', 'D', 'G', 'B', 'e']  # Da mais grave para a mais aguda
MIDI_CORDAS_SOLTAS = [40, 45, 50, 55, 59, 64]   # E2, A2, D3, G3, B3, E4
NUMERO_CASAS = 22  # Guitarra padrão (trastes 0 a 22)


def encontrar_posicoes_possiveis(midi_nota: int, max_casas: int = 22) -> List[Tuple[int, int]]:
    """
    Retorna todas as tuplas (indice_corda, casa) onde a nota MIDI pode ser tocada fisicamente.

    Parâmetros:
        midi_nota (int): Número MIDI da nota (ex: 40 para E2, 60 para C4).
        max_casas (int): Número máximo de casas disponíveis na escala.

    Retorna:
        List[Tuple[int, int]]: Lista de pares (corda_idx, casa), onde corda_idx vai de 0 (6ª) a 5 (1ª).
    """
    posicoes = []
    for corda_idx, midi_solta in enumerate(MIDI_CORDAS_SOLTAS):
        casa = midi_nota - midi_solta
        if 0 <= casa <= max_casas:
            posicoes.append((corda_idx, casa))
    return posicoes


def selecionar_melhor_posicao(
    posicoes_candidatas: List[Tuple[int, int]],
    posicao_anterior: Optional[Tuple[int, int]] = None
) -> Tuple[int, int]:
    """
    Seleciona a melhor posição física (corda, casa) minimizando o custo biomecânico
    do movimento da mão esquerda do músico no braço da guitarra.

    Heurística de Custo:
      - Se for a primeira nota (sem anterior): prefere casas mais baixas (posição aberta / casas 0 a 4).
      - Se houver nota anterior: custo = penalidade de mudança de casa + penalidade de salto entre cordas distantes.
        Custo = |casa_nova - casa_antiga| + 0.5 * |corda_nova - corda_antiga|
        Se for corda solta (casa 0), aplica bônus de conforto mecânico.
    """
    if not posicoes_candidatas:
        raise ValueError("Nenhuma posição candidata encontrada no braço para a nota.")

    if posicao_anterior is None:
        # Primeira nota: prefere a posição natural entre as casas 0 e 5
        def custo_inicial(pos: Tuple[int, int]) -> float:
            _, casa = pos
            return casa if casa <= 5 else casa * 1.5
        return min(posicoes_candidatas, key=custo_inicial)

    corda_ant, casa_ant = posicao_anterior

    def calcular_custo(pos: Tuple[int, int]) -> float:
        corda_nova, casa_nova = pos
        # Distância horizontal no braço (deslocamento longitudinal da mão)
        delta_casa = abs(casa_nova - casa_ant)
        # Distância vertical (pulo entre cordas)
        delta_corda = abs(corda_nova - corda_ant)

        custo = delta_casa + (0.5 * delta_corda)
        # Bônus para corda solta se o movimento não for muito distante
        if casa_nova == 0:
            custo -= 0.5
        return custo

    return min(posicoes_candidatas, key=calcular_custo)


def mapear_notas_para_tablatura(lista_notas: List[Dict]) -> List[Dict]:
    """
    Processa uma lista de notas detectadas e adiciona para cada nota os campos:
      - 'string_index': 0 a 5 (0 = 6ª corda Mi, 5 = 1ª corda mi)
      - 'string_name': 'E', 'A', 'D', 'G', 'B' ou 'e'
      - 'string_number_idmt': 1 a 6 (formato IDMT, 1 = 6ª corda, 6 = 1ª corda)
      - 'fret': número do traste (0 = solta)
      - 'all_candidates': lista de todas as outras posições físicas válidas
    """
    notas_com_tab = []
    pos_anterior = None

    for nota in lista_notas:
        midi_num = nota.get('midi_number')
        if midi_num is None:
            continue

        candidatos = encontrar_posicoes_possiveis(midi_num)
        if not candidatos:
            continue

        melhor_corda, melhor_casa = selecionar_melhor_posicao(candidatos, pos_anterior)
        pos_anterior = (melhor_corda, melhor_casa)

        registro = dict(nota)
        registro['string_index'] = melhor_corda
        registro['string_name'] = NOMES_CORDAS[melhor_corda]
        # Padrão IDMT: stringNumber 1 = 6ª corda (E2), 6 = 1ª corda (e4)
        registro['string_number_idmt'] = melhor_corda + 1
        registro['fret'] = melhor_casa
        registro['all_candidates'] = candidatos
        notas_com_tab.append(registro)

    return notas_com_tab


def renderizar_tablatura_ascii(notas_mapeadas: List[Dict], espaco_entre_notas: int = 3) -> str:
    """
    Gera a representação visual em texto (ASCII) das 6 cordas da guitarra
    com os trastes posicionados sequencialmente no tempo.

    Exemplo de saída:
    e|-----------------------------|
    B|-----------------------------|
    G|-----------------------------|
    D|-----------------------------|
    A|---------0-------------------|
    E|---0--3----------------------|
    """
    if not notas_mapeadas:
        return "Nenhuma nota disponível para renderizar tablatura."

    # Linhas de tablatura (da 1ª corda 'e' até a 6ª corda 'E')
    # Ordem visual tradicional: topo = 1ª corda (aguda), base = 6ª corda (grave)
    prefixos = ['e|--', 'B|--', 'G|--', 'D|--', 'A|--', 'E|--']
    linhas = {corda_idx: [prefixos[5 - corda_idx]] for corda_idx in range(6)}

    for nota in notas_mapeadas:
        corda_ativa = nota['string_index']
        casa = str(nota['fret'])
        largura_celula = max(len(casa), 2) + espaco_entre_notas

        for corda_idx in range(6):
            if corda_idx == corda_ativa:
                linhas[corda_idx].append(f"{casa}".ljust(largura_celula, '-'))
            else:
                linhas[corda_idx].append('-' * largura_celula)

    # Fecha as barras finais
    saida = []
    # Imprime da corda 5 (1ª corda, 'e') até a corda 0 (6ª corda, 'E')
    for corda_idx in range(5, -1, -1):
        saida.append("".join(linhas[corda_idx]) + "|")

    return "\n".join(saida)