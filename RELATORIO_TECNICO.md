# RELATÓRIO TÉCNICO DE DESENVOLVIMENTO (TCC 1)

**Projeto:** Sistema de Transcrição Automática de Áudio de Guitarra para Tablatura  
**Área:** Recuperação de Informação Musical (MIR), Processamento Digital de Sinais (DSP) e Visão Computacional  
**Repositório:** `tcc-transcricao-de-audio`  
**Escopo do Relatório:** Fundamentação e Implementação das Etapas 1 e 2  

---

## 1. RESUMO EXECUTIVO E CONTEXTO DO PROJETO

O objetivo central deste Trabalho de Conclusão de Curso (TCC) é construir uma pipeline computacional capaz de receber gravações em áudio monofônicas/polifônicas de guitarra e transcrevê-las automaticamente para o formato de tablatura.

A transcrição de guitarra apresenta dois desafios clássicos na literatura de *Music Information Retrieval* (MIR):
1. **O Problema da Série Harmônica (Pitch Tracking):** Uma única corda vibrando não produz apenas uma frequência pura (senoide), mas uma onda complexa contendo a Frequência Fundamental ($f_0$) somada a dezenas de sobretons múltiplos ($2f_0, 3f_0, 4f_0, \dots$). Algoritmos ingênuos interpretam esses harmônicos como notas adicionais, confundindo uma nota isolada com um acorde de 5 ou 6 notas.
2. **A Ambiguidade do Braço da Guitarra (*Fretboard Ambiguity*):** Diferente do piano — onde cada tecla produz uma nota única e exclusiva —, na guitarra a exata mesma nota (mesmo tom e oitava) pode ser executada em até 4 cordas e casas distintas.

Para solucionar a primeira metade desse desafio com máxima robustez acadêmica, optou-se por **uma abordagem híbrida pioneira**: converter o sinal de áudio em um espectrograma adaptado à escala musical (CQT) e tratá-lo com técnicas de **Visão Computacional e Morfologia Matemática**, transformando a detecção acústica em um problema de **detecção e rotulação de objetos geométricos conexos**.

---

## 2. ETAPA 1: PRÉ-PROCESSAMENTO E LIMPEZA ESPECTRAL

### 2.1. Escolha da Transformada: CQT (*Constant-Q Transform*) vs. FFT/STFT
* **Decisão Tomada:** Utilizou-se a Constant-Q Transform (`librosa.cqt`) em vez da Transformada Rápida de Fourier clássica (FFT).
* **Justificativa Teórica:**  
  A FFT divide o espectro eletroacústico em faixas de largura uniforme em Hertz (resolução linear). Contudo, a audição humana e a afinação musical ocidental são intrinsecamente logarítmicas. Nos registros graves, a distância entre semitons sucessivos é estreita (ex.: E2 a F2 distam apenas ~4.9 Hz), enquanto nos agudos a distância se alarga para centenas de Hertz.  
  A CQT mantém a razão $Q = \frac{f}{\Delta f}$ constante ao longo de todas as oitavas (Brown, 1991). Dessa forma, cada "linha" da matriz gerada corresponde rigorosamente a um semitom musical (uma casa do braço do instrumento), espelhando a física da escala cromática temperada de 12 semitons.

### 2.2. Calibração dos Parâmetros Críticos da CQT
* **Taxa de Amostragem (`sr = 22050 Hz`):**  
  *Justificativa:* Pelo Teorema da Amostragem de Nyquist-Shannon, a frequência máxima reproduzível sem aliasing é $f_{max} = \frac{sr}{2} = 11025\text{ Hz}$. Na guitarra elétrica e acústica, a fundamental da nota mais aguda raramente ultrapassa 1300 Hz (E6 na 24ª casa), e harmônicos superiores a 8000 Hz carregam energia inaudível ou ruído de palhetada. Reduzir a taxa de 44.100 Hz para 22.050 Hz corta o volume de tensores em 50%, acelerando o processamento sem qualquer perda musical.
* **Frequência Mínima (`fmin = E2 ≈ 82.41 Hz`):**  
  *Justificativa:* Na afinação padrão (E-A-D-G-B-e), a corda mais grave solta é o Mi grave (E2, 82.41 Hz). Qualquer oscilação abaixo desse valor constitui ruído subsônico de manuseio mecânico, batida na caixa ou vibração de 60 Hz da rede elétrica. O descarte na origem limpa a matriz antes de qualquer processamento.
* **Número de Bins (`n_bins = 84`):**  
  *Justificativa:* Como a CQT opera com 12 divisões por oitava, $84 / 12 = 7\text{ oitavas}$. Isso cobre de E2 (~82 Hz) até aproximadamente E9 (~10.500 Hz), englobando todo o braço do instrumento e seus harmônicos audíveis.
* **Passo de Janela (`hop_length = 512 amostras`):**  
  *Justificativa:* O passo temporal define a taxa de quadros (frames). Com $sr=22050$, cada coluna do espectrograma representa:
  $$\Delta t = \frac{512}{22050} \approx 0.0232\text{ segundos} \approx 23.2\text{ ms}$$
  Na execução mais veloz viável (ex.: semicolcheias a 180 BPM, com duração de ~83 ms), cada nota é representada por no mínimo 3 a 4 colunas consecutivas na matriz, garantindo precisão cronológica sem superdimensionar a memória.

### 2.3. Conversão para Escala Logarítmica de Intensidade (Decibéis)
A matriz de amplitude linear da CQT passa pela conversão:
$$\text{dB} = 20 \cdot \log_{10}\left(\frac{|X|}{\max(|X|)}\right)$$
O parâmetro `ref = np.max` normaliza o pico mais energético do áudio para $0\text{ dB}$, tornando todos os demais valores negativos. Isso respeita a Lei Psicofísica de Weber-Fechner (percepção humana de volume) e permite separar sinais úteis de caudas de decaimento.

### 2.4. Limiarização Rígida (*Hard Thresholding* / Noise Gate)
* **Decisão Tomada:** Limiar fixado em $-40\text{ dB}$ relativo ao pico.
* **Justificativa Acústica:**  
  Em gravações de guitarra, notas tocadas e seus harmônicos imediatos concentram-se tipicamente entre $-5\text{ dB}$ e $-30\text{ dB}$. A reverberação de ambiente (eco de sala) e o ruído térmico de amplificadores caem abaixo de $-45\text{ dB}$. O limiar de $-40\text{ dB}$ atua como um *Noise Gate* cirúrgico (Zölzer, 2011), zerando o fundo sem decapitar notas com dinâmica mais suave.

---

## 3. A TRANSIÇÃO PARA VISÃO COMPUTACIONAL: MÁSCARA BINÁRIA E MORFOLOGIA

### 3.1. Conceito da Máscara Binária Tempo-Frequência
Em vez de manipular destrutivamente os decibéis da matriz, o sistema gera uma matriz booleana independente:
$$M(f, t) = \begin{cases} 1, & \text{se } S(f, t) > -40\text{ dB} \\ 0, & \text{se } S(f, t) \le -40\text{ dB} \end{cases}$$
Essa técnica é derivada do conceito de *Ideal Binary Mask* (IBM) em processamento de fala e separação de fontes. O espectrograma vira uma imagem binária em preto e branco.

### 3.2. O Desafio dos Transientes e a Solução Morfológica Direcional
Durante os testes com notas isoladas (arquivo `som4.mp3`), detectou-se um problema crítico:
* **O Problema do "Pente":** O impacto físico da palheta na corda gera um estalo percussivo de banda larga instantâneo (uma coluna vertical que atravessa dezenas de frequências durante ~40 ms). Ao aplicar fechamento morfológico genérico, essa barra vertical servia como uma "ponte", fundindo a nota fundamental e todos os seus harmônicos em um único bloco monolítico conectado.
* **A Solução Morfológica Direcional (Operadores Sequenciais):**
  1. **Abertura Horizontal (`structure = np.ones((1, 5))`):**  
     A abertura realiza Erosão seguida de Dilatação. Um elemento estruturante $1 \times 5$ (1 semitom de altura $\times$ ~115 ms de largura) apaga qualquer mancha que não tenha sustentação temporal horizontal. Como o estalo percussivo vertical da palheta dura menos de 50 ms, ele é completamente eliminado, desacoplando os harmônicos horizontais antes de qualquer fusão.
  2. **Fechamento Horizontal (`structure = np.ones((1, 15))`):**  
     O fechamento realiza Dilatação seguida de Erosão. O elemento $1 \times 15$ (~345 ms) fecha pequenos vales de intensidade provocados pelo vibrato natural da mão do guitarrista, garantindo que uma nota longa não seja interpretada como múltiplas notas picotadas.

---

## 4. ETAPA 2: DETECÇÃO DE ONSETS, OFFSETS E PITCH TRACKING

Ao final da Etapa 1, a imagem binária contém "ilhas" brancas convexas e limpas. A Etapa 2 executa a extração musical através de componentes conexos.

### 4.1. Rotulação de Componentes Conexos (*Connected Components Labeling*)
Utiliza-se o algoritmo `scipy.ndimage.label` com vizinhança de 8 conectividades. Cada ilha de pixels recebe um identificador único.  
Por meio de `scipy.ndimage.find_objects`, extraem-se as coordenadas extremas de cada ilha (sua *Bounding Box*):
* `frame_min` e `frame_max` $\rightarrow$ Convertidos para tempo em segundos:
  $$t_{\text{onset}} = \frac{\text{frame\_min} \cdot \text{hop\_length}}{sr}, \quad t_{\text{offset}} = \frac{\text{frame\_max} \cdot \text{hop\_length}}{sr}$$
* `bin_min` e `bin_max` $\rightarrow$ Mapeados para notas e frequências em Hz.

### 4.2. Filtro de Área Mínima (Rejeição de Batidas Mecânicas)
Componentes com área inferior a $40\text{ pixels}$ são descartados. Esse limiar foi calibrado empiricamente para rejeitar cliques de apoio de dedos na madeira e ruídos transitórios de palhetada que tenham sobrevivido à morfologia.

### 4.3. Agrupamento Harmônico (*Harmonic Sieve*) e Identificação da Fundamental ($f_0$)
Para resolver o problema da série harmônica:
1. Os componentes válidos são agrupados por coincidência temporal de ataque:
   $$|t_{\text{onset}, A} - t_{\text{onset}, B}| \le 0.35\text{ segundos}$$
2. Componentes que nascem juntos e têm sobreposição temporal são associados ao mesmo "evento musical".
3. **Regra Acústica:** Dentro de um mesmo grupo de palhetada, o componente com **menor índice de frequência (bin)** é categorizado como a **Frequência Fundamental ($f_0$)**. Todos os componentes de frequência superior pertencentes àquele evento são arquivados como seus respectivos harmônicos acompanhantes e removidos da contagem de notas.

---

## 5. RESULTADOS EXPERIMENTAIS E VALIDAÇÃO

A pipeline integrada foi testada com sucesso sobre dois arquivos representativos da base de dados do projeto:

### Teste A: Três Notas Consecutivas (`som2.wav`)
* **Duração total:** 18.25 segundos
* **Componentes brutos na máscara:** 9 componentes
* **Notas Fundamentais identificadas após agrupamento harmônico:** 3 notas

| Evento | Nota Musical | Freq. Central ($f_0$) | Onset (Ataque) | Offset (Término) | Duração | Harmônicos Filtrados Agrupados |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Nota 1** | **E3** | 164.8 Hz | 0.42 s | 3.37 s | 2.95 s | E4 (329.6 Hz), B4 (493.9 Hz) |
| **Nota 2** | **G#3** | 207.7 Hz | 7.04 s | 7.36 s | 0.33 s | G#4 (415.3 Hz) |
| **Nota 3** | **C4** | 261.6 Hz | 12.65 s | 15.86 s | 3.20 s | C5 (523.3 Hz), G5 (784.0 Hz), E6 (1318.5 Hz) |

*Diagnóstico:* As três notas musicais foram identificadas com exatidão tonal e harmônica. Note-se a precisão do agrupamento: a nota C4 (261.6 Hz) teve 3 harmônicos (C5 a 523 Hz, G5 a 784 Hz e E6 a 1318 Hz) devidamente associados à sua emissão original.

### Teste B: Nota Única Sustentada com Alta Densidade Harmônica (`som4.mp3`)
* **Duração total:** 4.90 segundos
* **Execução:** Mi grave da 6ª corda solta
* **Notas Fundamentais identificadas:** 1 nota

| Evento | Nota Musical | Freq. Central ($f_0$) | Onset (Ataque) | Offset (Término) | Duração | Harmônicos Filtrados Agrupados |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Nota 1** | **E2** | 82.4 Hz | 0.95 s | 4.74 s | 3.78 s | D#3, A#3, D#4, C#5, D#6 |

*Diagnóstico:* A pipeline desacoplou com sucesso a barra de ataque do sustain, identificou o E2 (82.4 Hz) como a nota raiz e evitou qualquer falso positivo de acorde.

---

## 6. PRÓXIMOS PASSOS (ETAPA 3 - TRANSCRIÇÃO PARA TABLATURA)

Com a extração de $(t_{\text{onset}}, t_{\text{offset}}, f_0)$ concluída e validada, o projeto entra na Etapa 3:
1. **Resolução da Ambiguidade do Braço (*Fretboard Mapping*):**  
   Mapear cada nota $f_0$ para o par $(\text{corda}, \text{casa})$. Como a mesma nota existe em posições diferentes, será implementado um algoritmo de otimização de caminho (como Modelos Ocultos de Markov com decodificação de Viterbi ou algoritmo de Dijkstra), minimizando a distância física entre os deslocamentos da mão do músico.
2. **Exportação de Tablatura:**  
   Geração da tablatura final formatada em texto ASCII e em arquivo MusicXML para importação em softwares de notação musical (ex.: Guitar Pro / MuseScore).

---

## 7. REFERÊNCIAS BIBLIOGRÁFICAS

1. **BELLO, J. P. et al.** A Tutorial on Onset Detection in Music Signals. *IEEE Transactions on Speech and Audio Processing*, v. 13, n. 5, p. 1035–1047, 2005.
2. **BROWN, J. C.** Calculation of a constant Q spectral transform. *The Journal of the Acoustical Society of America*, v. 89, n. 1, p. 425–434, 1991.
3. **DE CHEVEIGNÉ, A.; KAWAHARA, H.** YIN, a fundamental frequency estimator for speech and music. *The Journal of the Acoustical Society of America*, v. 111, n. 4, p. 1917–1930, 2002.
4. **GONZALEZ, R. C.; WOODS, R. E.** *Processamento Digital de Imagens*. 3. ed. São Paulo: Pearson Prentice Hall, 2010. (Cap. 9: Morfologia Matemática).
5. **KLAPURI, A.** Multiple Fundamental Frequency Estimation based on Harmonicity and Spectral Smoothness. *IEEE Transactions on Speech and Audio Processing*, v. 11, n. 6, p. 804–816, 2003.
6. **OPPENHEIM, A. V.; SCHAFER, R. W.** *Discrete-Time Signal Processing*. 3. ed. Upper Saddle River: Prentice Hall, 2009.
7. **WANG, X. et al.** GuitarSet: A Dataset for Guitar Transcription. In: *International Society for Music Information Retrieval Conference (ISMIR)*, 2018.
8. **ZÖLZER, U.** *DAFX: Digital Audio Effects*. 2. ed. Chichester: John Wiley & Sons, 2011. (Cap. 5: Non-linear and Dynamic Range Control).