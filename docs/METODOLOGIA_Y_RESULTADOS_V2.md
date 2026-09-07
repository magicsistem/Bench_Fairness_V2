# Metodología y resultados de la tesis — V2

## Resumen ejecutivo

Este documento explica de forma autocontenida la metodología V2 definida por las decisiones D01–D60 de la guía canónica y presenta los resultados científicos obtenidos en CEDIA. La pregunta central es cómo se comportan distintos métodos de segmentación de lesiones dermatoscópicas y qué tan robustos son ante cambios controlados de tono de piel. La V2 separa estrictamente desarrollo, selección y evaluación final: `Training` se usa para desarrollar YOLOv7 mediante cinco folds; las 100 imágenes de `Validation` seleccionan el TOP-3 entre 15 checkpoints conservados más GrabCut; las 1000 imágenes de `Test` permanecen selladas hasta el `scientific freeze`; y MSKCC se usa exclusivamente para evaluar la recuperación del tono, no la exactitud de segmentación.

Los modelos seleccionados fueron **AViT**, **DeLightSAM-Dermoscopy** y **VM-UNet ISIC17**. En Test original, AViT obtuvo el mayor Jaccard umbralizado medio (0.7667), seguido de VM-UNet ISIC17 (0.7545) y DeLightSAM-Dermoscopy (0.7425). El experimento sintético MST mostró tres perfiles diferentes: AViT presentó degradaciones medias pequeñas pero persistentes; DeLightSAM se mantuvo cerca de cero en los tonos finales; y VM-UNet ISIC17 sufrió una caída pronunciada desde MST 06 hasta MST 10, llegando a un cambio medio de Jaccard de −0.3405 en MST 10. En MSKCC, VM-UNet ISIC17 produjo la mayor concordancia absoluta de ITA con el colorímetro (ICC=0.5145), aunque los errores siguieron siendo grandes. Estos resultados describen robustez del pipeline respecto del color observado o sintetizado; no demuestran fairness demográfica ni equivalencia clínica.

## Alcance, gobernanza y trazabilidad

La fuente normativa es `10_METHODOLOGY_V2_FROM_ZERO.md`. Este informe no introduce decisiones nuevas: resume D01–D60 y sus resultados. Todo entrenamiento, inferencia y generación de resultados científicos se ejecutó mediante Slurm en CEDIA, únicamente en `compute-0-2`, con `compute-0-1` excluido. No hubo revisión manual de imágenes. Los checkpoints de los 15 modelos históricos y GrabCut se conservaron; se repitieron las evaluaciones y no se heredaron predicciones, ranking ni TOP-3 V1.

| Elemento de trazabilidad | Valor verificado |
|---|---|
| Repositorio | `magicsistem/Bench_Fairness_V2` |
| Commit científico final | `56c1447b0f3d2cb2b3246910aac0b4fcf333eb93` |
| Commit que cierra G1–G8 | `0e61764577c2b2793e169fe8b64b8b530b362404` |
| Nodo científico | `compute-0-2` |
| Finalizador Slurm | job `25037`, `COMPLETED`, `ExitCode=0:0` |
| Resultado del finalizador | `V2_COMPLETE` |
| Bootstrap | 10 000 réplicas; semilla `20260828` |
| Revisión manual de imágenes | Ninguna |

## Metodología paso a paso

### Paso 1. Partición de datos y prevención de leakage

Se utilizó ISIC 2018 Task 1 [1], [18]. Las 2594 imágenes de `Training` se dividieron en cinco folds disjuntos. Cada imagen fue predicha por un detector que no la había visto durante entrenamiento, generando predicciones *out-of-fold* (OOF). Las 100 imágenes de `Validation` se reservaron para evaluar los 16 segmentadores y elegir el TOP-3. Las 1000 imágenes de `Test` se mantuvieron selladas hasta completar el `scientific freeze` [19], [34].

La caja de referencia de una máscara binaria de lesión \(M_{GT}\) fue:

\[
B_{GT}=(x_{\min},y_{\min},x_{\max},y_{\max}),
\]

\[
x_{\min}=\min\{x:M_{GT}(x,y)=1\},\qquad
x_{\max}=\max\{x:M_{GT}(x,y)=1\},
\]

con una definición análoga para \(y\). Los identificadores relacionados se mantuvieron agrupados cuando los metadatos lo permitieron; no se inventaron identidades faltantes.

### Paso 2. Entrenamiento del detector YOLOv7

YOLOv7 se eligió como localizador de lesión por su diseño *one-stage*, implementación pública y evidencia comparativa en dermatoscopia [2], [3]. Se utilizó una sola clase, `lesion`, entrada 640×640 con conservación de aspecto, checkpoint oficial `yolov7_training.pt`, ajuste completo de `backbone`, `neck` y `head`, minibatch físico 32 y acumulación de dos minibatches, es decir:

\[
B_{efectivo}=32\times 2=64.
\]

El optimizador fue SGD con Nesterov, \(lr_0=0.005\), momentum 0.900, `weight_decay=0.0005`, *warm-up* de tres epochs y calendario cosenoidal One-Cycle hasta \(lr_f=0.0005\). Cada fold se entrenó 300 epochs sin *early stopping*. El checkpoint se eligió con la *fitness* oficial [13]:

\[
F=0.1\,mAP@0.5+0.9\,mAP@0.5{:}0.95.
\]

Las augmentations, `ComputeLossOTA`, AutoAnchor y demás hiperparámetros siguieron la configuración oficial congelada. En inferencia se usaron confianza 0.25 y NMS IoU 0.45. Si la ejecución era válida pero no había caja, se registró `valid_no_detection` y se utilizó la imagen completa; un fallo técnico no se convirtió en una no-detección.

### Paso 3. Calibración OOF de la ROI

Para cada caja predicha OOF \(B_i^p\), de ancho \(w_i\) y alto \(h_i\), se midió el déficit relativo necesario para cubrir la caja GT:

\[
r_{L,i}=\max\left(0,\frac{x^p_{\min}-x^{GT}_{\min}}{w_i}\right),\quad
r_{R,i}=\max\left(0,\frac{x^{GT}_{\max}-x^p_{\max}}{w_i}\right),
\]

\[
r_{T,i}=\max\left(0,\frac{y^p_{\min}-y^{GT}_{\min}}{h_i}\right),\quad
r_{B,i}=\max\left(0,\frac{y^{GT}_{\max}-y^p_{\max}}{h_i}\right).
\]

El margen individual y el margen global fueron:

\[
r_i=\max(r_{L,i},r_{R,i},r_{T,i},r_{B,i}),\qquad
m^*=Q_{0.95}(r_1,\ldots,r_N).
\]

También se cuantificó la fracción de lesión contenida por la ROI:

\[
C_i(m)=\frac{|M_i^{GT}\cap ROI_i(m)|}{|M_i^{GT}|}.
\]

Las no-detecciones se excluyeron de la estimación del percentil y se informaron por separado [4], [5].

### Paso 4. Inferencia de los 16 segmentadores en Validation

Los candidatos fueron 15 checkpoints conservados más GrabCut. Todos recibieron el mismo `roi_rgb` original, sin FOV, eliminación de pelo, *inpainting* ni transformación de color. Cada adaptador mantuvo su tamaño, normalización y regla de binarización nativos. La salida binaria de ROI se restauró a coordenadas de imagen completa y no recibió posprocesamiento morfológico común. Las máscaras vacías o llenas fueron predicciones científicas degeneradas; los fallos técnicos se corrigieron y reejecutaron [19], [20].

El Jaccard continuo y Dice fueron:

\[
J=\frac{TP}{TP+FP+FN},\qquad
Dice=\frac{2TP}{2TP+FP+FN}.
\]

La métrica primaria oficial de ISIC 2018 fue el Jaccard umbralizado [1], [18]:

\[
TJ_i=\begin{cases}
0,&J_i<0.65,\\
J_i,&J_i\ge 0.65,
\end{cases}
\qquad
\overline{TJ}=\frac{1}{N}\sum_{i=1}^{N}TJ_i.
\]

Se conservaron Jaccard, Dice, sensibilidad, especificidad, precisión, Boundary F1 y HD95 normalizado como métricas secundarias.

### Paso 5. Ranking, incertidumbre y TOP-3

Solo fueron elegibles los métodos con cobertura completa de las 100 imágenes de `Validation`. En cada una de 10 000 réplicas bootstrap se remuestrearon imágenes completas con reemplazo y se utilizó el mismo vector de índices para todos los métodos. Para una métrica \(g\), la réplica \(b\) fue:

\[
\hat g^{(b)}=\frac{1}{N}\sum_{j=1}^{N}g_{I_j^{(b)}},
\qquad I_j^{(b)}\sim\text{Uniforme}\{1,\ldots,N\}.
\]

Los percentiles 2.5 y 97.5 de las 10 000 réplicas formaron el IC95%. El orden puntual siguió \(\overline{TJ}\); los desempates usaron Jaccard, Dice, Boundary F1 y menor HD95, en ese orden. Las distribuciones de posiciones se informaron porque los rankings biomédicos pueden depender fuertemente de los casos y de la métrica [21], [22].

### Paso 6. Construcción de `clean_skin_mask`

Cada método del TOP-3 produjo su propia máscara de piel limpia. No se fusionaron segmentadores. En la ROI, la piel candidata fue:

\[
M_{clean,m}=M_{ROI}\land \neg dilate(M_{lesion,m},r^*)
\land \neg M_{hair}\land \neg M_{highlight}.
\]

El detector morfológico de pelo se utilizó solo como exclusión colorimétrica, nunca para modificar la entrada del segmentador. Esta elección evita tratar un píxel sintetizado mediante *inpainting* como una medición real, aunque reconoce las limitaciones de los detectores morfológicos frente a pelo claro y falsos positivos [6]–[12]. Una cobertura de pelo mayor que 0.18 se conservó y marcó como alta.

El margen de lesión común se calibró como el Q95 de la distancia unilateral normalizada por el lado menor de la ROI. Las máscaras exactamente vacías o llenas produjeron colorimetría no disponible. Además, se exigió:

\[
N_{clean}\ge \max\left(256,\left\lceil0.005\,H_{ROI}W_{ROI}\right\rceil\right).
\]

Los reflejos saturados se excluyeron cuando:

\[
\max(R,G,B)\ge248
\quad\land\quad
\max(R,G,B)-\min(R,G,B)\le22.
\]

No se amplió el soporte, no se imputaron píxeles y no se revisaron imágenes manualmente.

### Paso 7. Conversión colorimétrica e ITA

Los JPEG se interpretaron como sRGB IEC 61966-2-1 [28]. Para cada canal normalizado \(c_s\in[0,1]\), se aplicó la linealización:

\[
c=\begin{cases}
c_s/12.92,&c_s\le0.04045,\\
\left(\frac{c_s+0.055}{1.055}\right)^{2.4},&c_s>0.04045.
\end{cases}
\]

El RGB lineal se convirtió a XYZ D65 en `float64`:

\[
\begin{bmatrix}X\\Y\\Z\end{bmatrix}
=100
\begin{bmatrix}
0.4124564&0.3575761&0.1804375\\
0.2126729&0.7151522&0.0721750\\
0.0193339&0.1191920&0.9503041
\end{bmatrix}
\begin{bmatrix}R\\G\\B\end{bmatrix}.
\]

Con el blanco D65, \(f(t)=t^{1/3}\) si \(t>(6/29)^3\), y \(f(t)=t/[3(6/29)^2]+4/29\) en otro caso, se obtuvo CIELAB [24], [29]:

\[
L^*=116f(Y/Y_n)-16,\quad
a^*=500[f(X/X_n)-f(Y/Y_n)],
\]

\[
b^*=200[f(Y/Y_n)-f(Z/Z_n)].
\]

El Individual Typology Angle se mantuvo como variable continua:

\[
ITA=\frac{180}{\pi}\operatorname{atan2}(L^*-50,b^*).
\]

No se transformó automáticamente ITA en Fitzpatrick ni MST. Se conservaron también mediana RGB, media RGB recortada 10–90 y mediana CIELAB [25], [27].

### Paso 8. `Scientific freeze` y apertura única de Test

Antes de materializar Test se congelaron commit, código, configuraciones, manifiestos, checkpoints, contenedor, dependencias, TOP-3, márgenes, semillas, métricas, paleta y análisis en `scientific_freeze.json` y un tag Git anotado. El verificador exigió repositorio limpio/publicado y ausencia de uso previo de Test. Después del freeze solo se permitieron reanudaciones técnicas con los mismos hashes [31], [34].

La apertura ocurrió una sola vez en CEDIA. Un DAG Slurm con dependencias `afterok` ejecutó inferencia ORIGINAL, métricas, síntesis MST, YOLOv7 por condición, TOP-3 por condición y análisis. La GT se utilizó únicamente en evaluación y en el soporte autorizado de síntesis; nunca se entregó a YOLOv7 ni a los segmentadores.

### Paso 9. Evaluación del Test original

YOLOv7 se ejecutó sobre las 1000 imágenes completas. Cada TOP-3 recibió la ROI dinámica o la imagen completa si correspondía `valid_no_detection`. Las máscaras se restauraron a dimensiones originales y se compararon con la GT. Además se estimó un ITA independiente de los segmentadores usando GT dilatada, pelo y reflejos, y se calculó Spearman entre ITA y Jaccard, Dice y Boundary F1. Esta relación se interpretó como robustez frente al ITA observado, no como fairness demográfica [23], [31].

### Paso 10. Síntesis MST de imagen completa

La paleta MST 01–10 se definió en un único artefacto sRGB protegido por hash. Para cada imagen original se estimó una mediana Lab sobre soporte independiente de los segmentadores:

\[
M_{support}=M_{imagen}\land\neg dilate(M_{GT},r^*)
\land\neg M_{hair}\land\neg M_{highlight}.
\]

Se exigió el mismo mínimo de soporte, ahora respecto del área completa. Si no se cumplía, las diez condiciones se registraban `unavailable` sin lanzar excepción. Para el tono \(k\):

\[
\Delta Lab_k=Lab_{target,k}-Lab_{source},
\]

\[
Lab'_k(x,y)=Lab_{original}(x,y)+\Delta Lab_k
\quad\forall(x,y)\text{ de la imagen completa}.
\]

Después se convirtió nuevamente a sRGB, se aplicó *clipping* al dominio válido y se guardó PNG RGB de 8 bits, sin pérdida, con `compress_level=6`. Se verificaron dimensiones, GT idéntica, ausencia de NaN/Inf, hashes de píxeles decodificados y correspondencia fuente–condición. CIEDE2000 se calculó según Sharma, Wu y Dalal [32].

### Paso 11. Evaluación end-to-end de cada condición MST

YOLOv7 se volvió a ejecutar independientemente en cada imagen MST completa. Cada segmentador recibió solo la ROI propia de esa condición, o la imagen completa ante no-detección válida. La variable primaria fue:

\[
\Delta J_{i,m,k}=J_{i,m,k}-J_{i,m,ORIGINAL}.
\]

Cada tono se mantuvo separado. Se calcularon media, mediana, IQR e IC95% por bootstrap pareado, además de la peor caída por fuente:

\[
\Delta J^{worst}_{i,m}=\min_{k\in\{1,\ldots,10\}}\Delta J_{i,m,k}.
\]

No se hicieron pruebas de hipótesis ni valores \(p\) para MST, porque las condiciones son medidas repetidas sintéticas y no observaciones demográficas independientes.

### Paso 12. Censo y procesamiento MSKCC

Se censaron las 4879 imágenes públicas de MSKCC y se conservaron todas las tomas disponibles. La jerarquía fue `patient_id → sitio/lesión → modo de adquisición → imagen`. Los valores de referencia ausentes quedaron `NA`; no se imputaron ni se seleccionaron casos según las salidas. MSKCC se usó solo para concordancia cromática, porque no existe una GT de lesión autorizada para validar segmentación [33], [34].

YOLOv7 produjo una ROI dinámica por imagen. Si no detectó, se aplicó el mismo fallback automático a imagen completa. Cada TOP-3 generó su máscara de lesión, `clean_skin_mask` y variables cromáticas de forma independiente.

### Paso 13. Concordancia MSKCC

La medida primaria fue ICC de acuerdo absoluto entre el ITA del pipeline y el ITA del colorímetro, con bootstrap por paciente. En forma general, para un modelo de dos vías de acuerdo absoluto:

\[
ICC(A,1)=\frac{MS_R-MS_E}
{MS_R+(k-1)MS_E+\frac{k}{n}(MS_C-MS_E)},
\]

donde \(MS_R\), \(MS_C\) y \(MS_E\) son cuadrados medios de filas, columnas y error; \(n\) es el número de unidades y \(k\) el número de mediciones. Cada réplica remuestreó pacientes completos.

Los errores secundarios fueron:

\[
Bias=\frac{1}{N}\sum_i(ITA_{pipe,i}-ITA_{ref,i}),
\]

\[
MAE=\frac{1}{N}\sum_i|ITA_{pipe,i}-ITA_{ref,i}|,
\quad
RMSE=\sqrt{\frac{1}{N}\sum_i(ITA_{pipe,i}-ITA_{ref,i})^2}.
\]

Los límites de Bland–Altman fueron [35]:

\[
LoA=Bias\pm1.96\,SD(ITA_{pipe}-ITA_{ref}).
\]

Para la referencia MST ordinal se usaron Kendall \(\tau_b\) y Spearman \(\rho\), sin convertir ITA a una categoría MST [33], [36], [37].

### Paso 14. Reanudación, hashes y cierre

Las etapas publicaron manifiestos atómicamente. Las inferencias MST mantuvieron un `progress.jsonl` por método y solo reutilizaron una máscara si coincidían método, ROI, procedencia y SHA-256. Las variantes PNG existentes se recalcularon y reutilizaron únicamente si coincidían dimensiones y hash de píxeles decodificados. Los fallos quedaron en la bitácora; no se borraron checkpoints, logs ni ledger. El job `25037` verificó los gates detector, selección, freeze, MST, MSKCC y procedencia y emitió `V2_COMPLETE`.

## Resultados reales

### Tabla general del pipeline

| Etapa | Universo/resultado | Evidencia principal |
|---|---:|---|
| ISIC Training | 2594 imágenes; 5 folds OOF | `roi_margin.json` |
| OOF YOLOv7 | 2551 detecciones válidas; 43 no-detecciones | margen ROI \(m^*=0.122540\) |
| Validation | 100 imágenes; 16 métodos completos | `top3.json` |
| TOP-3 V2 | AViT, DeLightSAM-Dermoscopy, VM-UNet ISIC17 | selección D28–D32 |
| Margen de piel D35 | 300 observaciones método–imagen | \(r^*=0.141842\) |
| Test original | 1000 imágenes por cada TOP-3 | 983 detecciones y 17 fallbacks |
| Síntesis MST | 975 fuentes disponibles × 10 tonos = 9750 variantes | 25 fuentes/250 condiciones `unavailable` |
| Segmentación MST | 9750 resultados por cada TOP-3 | 29 250 inferencias completas |
| MSKCC | 4879 imágenes censadas | 2825 detecciones; 2054 fallbacks |
| Cierre | G1–G8 satisfechos | job 25037, `V2_COMPLETE` |

### Entrenamiento YOLOv7

La tabla muestra el epoch con mayor fitness oficial, no el último epoch. Todos los entrenamientos completaron 300 epochs.

| Entrenamiento | Mejor epoch | Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 | Fitness |
|---|---:|---:|---:|---:|---:|---:|
| Fold 0 | 87 | 0.9420 | 0.9383 | 0.9649 | 0.7038 | 0.7299 |
| Fold 1 | 128 | 0.9318 | 0.9209 | 0.9600 | 0.7157 | 0.7401 |
| Fold 2 | 130 | 0.9229 | 0.9228 | 0.9588 | 0.6850 | 0.7124 |
| Fold 3 | 230 | 0.9419 | 0.9363 | 0.9638 | 0.7051 | 0.7310 |
| Fold 4 | 140 | 0.9474 | 0.9054 | 0.9613 | 0.7011 | 0.7271 |
| Reajuste final | 70 | 0.9223 | 0.9499 | 0.9504 | 0.6464 | 0.6768 |

### Ranking completo de los 16 métodos en Validation

Todas las medias corresponden a \(N=100\). `P(TOP-3)` es la proporción de réplicas bootstrap en que el método ocupó una de las tres primeras posiciones.

| Rank | Método | Jaccard umbralizado | Jaccard | Dice | Boundary F1 | HD95 norm. | P(TOP-3) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | AViT | 0.7787 | 0.8189 | 0.8861 | 0.5204 | 0.0558 | 0.9006 |
| 2 | DeLightSAM-Dermoscopy | 0.7673 | 0.8076 | 0.8788 | 0.4431 | 0.0569 | 0.6797 |
| 3 | VM-UNet ISIC17 | 0.7666 | 0.8047 | 0.8742 | 0.5462 | 0.0598 | 0.6331 |
| 4 | VM-UNet ISIC18 | 0.7664 | 0.8072 | 0.8777 | 0.5043 | 0.0600 | 0.5888 |
| 5 | UNet++ ISIC2018 | 0.7483 | 0.7965 | 0.8736 | 0.4457 | 0.0668 | 0.0847 |
| 6 | U-Net ISIC2018 | 0.7429 | 0.8009 | 0.8751 | 0.4789 | 0.0583 | 0.0261 |
| 7 | U-Net ResNet34 ISIC2018 | 0.7366 | 0.7802 | 0.8623 | 0.3965 | 0.0682 | 0.0255 |
| 8 | UltraLight VM-UNet | 0.7308 | 0.7880 | 0.8590 | 0.5631 | 0.0620 | 0.0602 |
| 9 | Attention U-Net ISIC2018 | 0.7138 | 0.7795 | 0.8585 | 0.4611 | 0.0612 | 0.0013 |
| 10 | SegFormer ISIC2018 | 0.6756 | 0.7468 | 0.8352 | 0.3345 | 0.0705 | 0.0000 |
| 11 | Theodore U-Net ISIC2018 | 0.6501 | 0.7227 | 0.8107 | 0.3160 | 0.0709 | 0.0000 |
| 12 | GrabCut | 0.6426 | 0.7519 | 0.8372 | 0.4265 | 0.0805 | 0.0000 |
| 13 | BA-Transformer | 0.5813 | 0.6829 | 0.7684 | 0.4169 | 0.0833 | 0.0000 |
| 14 | Theodore Inception ISIC2018 | 0.4309 | 0.6105 | 0.7170 | 0.3520 | 0.1087 | 0.0000 |
| 15 | SkinMamba ISIC17 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | 0.0000 |
| 16 | SkinMamba ISIC18 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | 0.0000 |

### TOP-3 en Test original

| Método | Jaccard umbralizado | Jaccard | Dice | Boundary F1 | Tiempo medio por imagen |
|---|---:|---:|---:|---:|---:|
| AViT | 0.7667 | 0.8092 | 0.8806 | 0.4989 | 2.573 s |
| DeLightSAM-Dermoscopy | 0.7425 | 0.7914 | 0.8691 | 0.4357 | 2.368 s |
| VM-UNet ISIC17 | 0.7545 | 0.7983 | 0.8694 | 0.5293 | 4.264 s |

El análisis de robustez frente al ITA continuo tuvo \(N=966\) pares disponibles por método:

| Método | \(\rho\) ITA–Jaccard (IC95%) | \(\rho\) ITA–Dice (IC95%) | \(\rho\) ITA–Boundary F1 (IC95%) |
|---|---:|---:|---:|
| AViT | 0.0479 [−0.0178, 0.1128] | 0.0479 [−0.0178, 0.1128] | 0.0621 [−0.0016, 0.1281] |
| DeLightSAM-Dermoscopy | 0.0022 [−0.0642, 0.0693] | 0.0022 [−0.0642, 0.0693] | 0.0004 [−0.0636, 0.0633] |
| VM-UNet ISIC17 | 0.1154 [0.0502, 0.1794] | 0.1154 [0.0502, 0.1794] | 0.1248 [0.0603, 0.1879] |

### Sensibilidad MST end-to-end

Cada celda de método muestra la media de \(\Delta J\) y su IC95% bootstrap. Todas las condiciones tienen \(N=975\) fuentes disponibles. La tasa de detección corresponde a YOLOv7 y es común a los tres segmentadores porque el detector se ejecutó una vez por condición completa.

| Condición | Detección YOLOv7 | AViT \(\Delta J\) | DeLightSAM \(\Delta J\) | VM-UNet ISIC17 \(\Delta J\) |
|---|---:|---:|---:|---:|
| MST 01 | 0.9641 | −0.0229 [−0.0315, −0.0148] | −0.0338 [−0.0431, −0.0249] | −0.0404 [−0.0508, −0.0299] |
| MST 02 | 0.9723 | −0.0178 [−0.0257, −0.0103] | −0.0305 [−0.0389, −0.0223] | −0.0397 [−0.0496, −0.0301] |
| MST 03 | 0.9785 | −0.0148 [−0.0223, −0.0074] | −0.0310 [−0.0391, −0.0232] | −0.0396 [−0.0494, −0.0300] |
| MST 04 | 0.9815 | −0.0125 [−0.0194, −0.0058] | −0.0222 [−0.0297, −0.0150] | −0.0378 [−0.0475, −0.0286] |
| MST 05 | 0.9795 | −0.0137 [−0.0200, −0.0077] | −0.0149 [−0.0218, −0.0084] | −0.0404 [−0.0500, −0.0313] |
| MST 06 | 0.9846 | −0.0077 [−0.0139, −0.0019] | −0.0060 [−0.0123, 0.0001] | −0.1487 [−0.1638, −0.1341] |
| MST 07 | 0.9897 | −0.0067 [−0.0128, −0.0005] | −0.0006 [−0.0070, 0.0056] | −0.1377 [−0.1531, −0.1230] |
| MST 08 | 0.9897 | −0.0148 [−0.0210, −0.0083] | 0.0057 [−0.0006, 0.0123] | −0.1518 [−0.1698, −0.1344] |
| MST 09 | 0.9959 | −0.0256 [−0.0327, −0.0184] | 0.0056 [−0.0019, 0.0132] | −0.2604 [−0.2820, −0.2391] |
| MST 10 | 0.9990 | −0.0266 [−0.0346, −0.0182] | 0.0037 [−0.0052, 0.0123] | −0.3405 [−0.3644, −0.3175] |

### Concordancia cromática MSKCC

Los denominadores difieren porque D36–D37 conservan como `NA` los casos sin soporte válido y porque no todas las imágenes tienen la referencia requerida. Los IC95% del ICC remuestrean pacientes completos.

| Método | N continuo | Pacientes | ICC absoluto (IC95%) | Bias ITA | MAE | RMSE | Bland–Altman LoA |
|---|---:|---:|---:|---:|---:|---:|---:|
| AViT | 1168 | 46 | 0.3357 [0.2367, 0.4247] | −4.0891 | 60.2784 | 75.7678 | [−152.4409, 144.2628] |
| DeLightSAM-Dermoscopy | 699 | 46 | 0.4490 [0.3590, 0.5216] | 14.7545 | 47.9649 | 61.3627 | [−102.0714, 131.5804] |
| VM-UNet ISIC17 | 1396 | 46 | 0.5145 [0.4146, 0.5894] | 11.5159 | 47.6528 | 59.8357 | [−103.6108, 126.6426] |

| Método | N con MST | Kendall \(\tau_b\) | Spearman \(\rho\) |
|---|---:|---:|---:|
| AViT | 3735 | −0.5500 | −0.6793 |
| DeLightSAM-Dermoscopy | 3092 | −0.6237 | −0.7786 |
| VM-UNet ISIC17 | 3873 | −0.6251 | −0.7770 |

### Integridad de los artefactos finales

| Artefacto | SHA-256 |
|---|---|
| `results/test_mst/analysis.json` | `9ec78ea56d2bc7e4315cc6fc70a9584fc7096c7173233175c208329a2fe59786` |
| `artifacts/mskcc/census.json` | `d2c1b142755f8debd856d26dd40c95d82e32e813b0f314d5647c4f914bd9fa6a` |
| `artifacts/mskcc/rois.json` | `d95fdb9bb331f7cadc82d501790bc48cda9b5c85222f1ec149c5eb5ef1a2fe17` |
| `results/mskcc_analysis.json` | `4d3559ecaa93655f9298094bd6e2ddfc1cb9e8c0044842677f5e9def4177e542` |
| `artifacts/final/provenance.json` | `581c1f26126477104e91dce592b2a45b093bd2bfcf93a87ca81feb2eb46872e5` |

## Interpretación de los resultados

### Selección y desempeño general

AViT fue el método con mejor desempeño primario tanto en Validation como en Test original. En Validation superó por 0.0114 puntos de Jaccard umbralizado a DeLightSAM y por 0.0121 a VM-UNet ISIC17. Sin embargo, la incertidumbre del ranking muestra que las posiciones cercanas no son rígidas: DeLightSAM y VM-UNet ISIC17 tuvieron probabilidades bootstrap TOP-3 de 0.6797 y 0.6331, mientras VM-UNet ISIC18, cuarto por una diferencia de solo 0.00018 en la métrica primaria, alcanzó 0.5888. Por ello, el TOP-3 debe entenderse como una selección operativa predeclarada sobre 100 imágenes, no como una prueba de superioridad universal [21], [22].

En Test original se conservó el TOP-3 congelado, aunque el orden interno cambió: VM-UNet ISIC17 pasó al segundo lugar por Jaccard umbralizado y DeLightSAM al tercero. Esto es precisamente el comportamiento esperado de un diseño honesto: Test estima generalización y no se usa para volver a seleccionar. AViT mantuvo la primera posición y VM-UNet produjo el mejor Boundary F1, lo que confirma que región y frontera capturan propiedades distintas [20].

Los dos modelos SkinMamba produjeron máscaras degeneradas con métricas cero en Validation. Se conservaron como resultados científicos bajo D26–D30 y no se ocultaron ni transformaron en fallos técnicos. GrabCut alcanzó Jaccard continuo 0.7519, pero el umbral oficial penalizó más sus casos deficientes y redujo su métrica primaria a 0.6426.

### Robustez respecto del ITA estimado en Test

AViT y DeLightSAM mostraron asociaciones prácticamente nulas entre ITA y las métricas de región; sus IC95% incluyeron cero. VM-UNet ISIC17 presentó asociaciones positivas pequeñas, con IC95% por encima de cero. Esto significa que, dentro de estas imágenes y bajo este estimador de tono, VM-UNet tendió a rendir algo mejor conforme aumentó ITA. No establece causalidad y tampoco identifica una diferencia por raza, etnia o fototipo: ISIC 2018 no proporciona esos atributos legítimos y el ITA proviene del color observado en cada imagen [23], [31].

### Efecto de las transformaciones MST

El detector no colapsó ante las transformaciones. Su tasa de detección varió de 96.41% en MST 01 a 99.90% en MST 10, y las no-detecciones válidas se procesaron con la imagen completa. Por tanto, las grandes caídas de VM-UNet ISIC17 en MST 06–10 no pueden explicarse únicamente por pérdida generalizada de detecciones; reflejan una sensibilidad end-to-end dominada por la respuesta del segmentador y, en algunos casos, por cambios de ROI.

AViT tuvo cambios medios negativos en todos los tonos, pero de magnitud moderada: entre −0.0067 y −0.0266. DeLightSAM se degradó en MST 01–06 y quedó cerca de cero en MST 07–10; sus intervalos finales incluyen cero, por lo que la muestra es compatible tanto con una variación pequeña negativa como positiva. VM-UNet ISIC17 fue estable hasta MST 05 en comparación con su caída posterior, pero descendió a −0.1487 en MST 06 y a −0.3405 en MST 10. Este patrón demuestra que una puntuación alta en imágenes originales no garantiza invariancia frente a cambios cromáticos controlados.

Las 25 fuentes sin soporte suficiente no se forzaron dentro del experimento. Sus 250 variantes se marcaron `unavailable`; las 975 fuentes restantes formaron bloques completos de diez tonos. Esta exclusión automática preserva el pareamiento y evita fabricar color cuando el fondo disponible no cumple el mínimo D59.

El experimento MST altera toda la imagen, incluida la lesión, y vuelve a ejecutar el detector y el segmentador. Por eso evalúa sensibilidad del sistema completo a una intervención cromática sintética; no reproduce pigmentación humana real, adquisición clínica, iluminación, textura ni distribución demográfica. Sus resultados no deben titularse como una medición de fairness clínica [23]–[25], [31].

### Recuperación de tono en MSKCC

VM-UNet ISIC17 logró la mayor concordancia absoluta con el colorímetro y los menores MAE/RMSE de los tres, seguido de DeLightSAM. AViT, pese a ser el mejor segmentador en ISIC, produjo el ICC cromático más bajo y el mayor error. Esto indica que la calidad de segmentación y la calidad de la región de piel usada para colorimetría son objetivos relacionados pero no equivalentes.

Ningún ICC se acercó a 1 y los límites de Bland–Altman fueron amplios. El resultado concuerda con la advertencia de que color extraído de dermatoscopía depende del dispositivo, iluminación y modo de captura, y no es intercambiable automáticamente con una medición instrumental [33], [35]. La mayor disponibilidad de VM-UNet (1396 observaciones continuas) también muestra que su geometría dejó soporte válido con mayor frecuencia que DeLightSAM (699); los denominadores deben acompañar siempre las métricas.

Las asociaciones MST–ITA fueron negativas y fuertes en magnitud relativa. El signo es coherente con el orden de la escala: tonos MST numéricamente más altos corresponden a menor ITA. Esto es asociación ordinal, no exactitud categórica. No se predijo MST desde ITA y no se calcularon categorías artificiales [33], [37].

### Conclusión integrada

La V2 muestra que AViT ofrece el mejor desempeño regional general en ISIC 2018 y una sensibilidad MST moderada; DeLightSAM ofrece la mayor estabilidad media en la mitad final de la escala sintética; y VM-UNet ISIC17 combina buen desempeño original y la mejor concordancia cromática MSKCC, pero exhibe una fragilidad cromática sintética muy marcada. Por tanto, no existe un único modelo dominante en todas las dimensiones. La conclusión defendible es que desempeño de segmentación, estabilidad ante intervención cromática y recuperación del tono son ejes distintos y deben reportarse separadamente.

## Limitaciones

- `Validation` contiene solo 100 imágenes; por ello se muestran distribuciones bootstrap y no únicamente posiciones puntuales.
- Algunos checkpoints públicos tienen procedencia de entrenamiento incompleta. Se conservaron por decisión D57 y esta incertidumbre no se presenta como ausencia demostrada de leakage.
- ISIC 2018 no contiene variables demográficas legítimas; los análisis de ITA y MST no prueban fairness demográfica.
- MST es una intervención determinista de color sobre imágenes completas. No reproduce biología, iluminación, sensor, textura ni contexto clínico.
- MSKCC no aporta GT de lesión para esta tarea; solo permite evaluar colorimetría y asociación ordinal.
- Los valores MSKCC tienen denominadores distintos por referencias ausentes y por los controles de disponibilidad D36–D37.
- El detector de pelo es morfológico y puede perder pelo claro o excluir estructuras pigmentarias similares a pelo [8], [10], [11].
- La concordancia de una medida derivada de JPEG con un colorímetro está limitada por adquisición, dispositivo y procesamiento [25], [33].

## Referencias

[1] N. C. F. Codella *et al.*, “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC),” *arXiv:1902.03368*, 2019, doi: 10.48550/arXiv.1902.03368.

[2] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors,” in *Proc. IEEE/CVF CVPR*, 2023, pp. 7464–7475, doi: 10.1109/CVPR52729.2023.00721.

[3] N. A. AlSadhan, S. A. Alamri, M. M. Ben Ismail, and O. Bchir, “Skin Cancer Recognition Using Unified Deep Convolutional Neural Networks,” *Cancers*, vol. 16, no. 7, Art. 1246, 2024, doi: 10.3390/cancers16071246.

[4] M. E. Celebi, G. Schaefer, H. Iyatomi, and W. V. Stoecker, “Approximate Lesion Localization in Dermoscopy Images,” *Skin Research and Technology*, vol. 15, no. 3, pp. 314–322, 2009, doi: 10.1111/j.1600-0846.2009.00357.x.

[5] T. Nagaoka, “Improved Skin Lesion Segmentation in Dermoscopic Images Using Object Detection and Semantic Segmentation,” *Clinical, Cosmetic and Investigational Dermatology*, vol. 18, pp. 1191–1198, 2025, doi: 10.2147/CCID.S518751.

[6] T. Lee, V. T. Y. Ng, R. Gallagher, A. Coldman, and D. McLean, “DullRazor: A Software Approach to Hair Removal from Images,” *Computers in Biology and Medicine*, vol. 27, no. 6, pp. 533–543, 1997, doi: 10.1016/S0010-4825(97)00020-6.

[7] M. Hasan *et al.*, “Skin Lesion Segmentation from Dermoscopic Images Using Convolutional Neural Network,” *Sensors*, vol. 20, no. 6, Art. 1601, 2020, doi: 10.3390/s20061601.

[8] W. Li, A. N. J. Raj, T. Tjahjadi, and Z. Zhuang, “Digital Hair Removal by Deep Learning for Skin Lesion Segmentation,” *Pattern Recognition*, vol. 117, Art. 107994, 2021, doi: 10.1016/j.patcog.2021.107994.

[9] S. Joseph and O. O. Olugbara, “Preprocessing Effects on Performance of Skin Lesion Saliency Segmentation,” *Diagnostics*, vol. 12, no. 2, Art. 344, 2022, doi: 10.3390/diagnostics12020344.

[10] R. Kasmi, W. V. Stoecker, and R. J. Stanley, “SharpRazor: Automatic Removal of Hair and Ruler Marks from Dermoscopy Images,” *Skin Research and Technology*, 2023, doi: 10.1111/srt.13203.

[11] S. I. Hossain *et al.*, “A Skin Lesion Hair Mask Dataset with Fine-Grained Annotations,” *Data in Brief*, vol. 48, Art. 109249, 2023, doi: 10.1016/j.dib.2023.109249.

[12] S. I. Hossain *et al.*, “A Skin Lesion Hair Mask Dataset with Fine-Grained Annotations,” Mendeley Data, Version 2, 2023, doi: 10.17632/j5ywpd2p27.2.

[13] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “Official YOLOv7,” GitHub repository. [Online]. Available: https://github.com/WongKinYiu/yolov7.

[14] J. Redmon, “YOLO: Real-Time Object Detection,” Darknet. [Online]. Available: https://pjreddie.com/darknet/yolo/.

[15] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, “How Transferable Are Features in Deep Neural Networks?,” in *Advances in Neural Information Processing Systems*, vol. 27, 2014, pp. 3320–3328.

[16] N. Tajbakhsh *et al.*, “Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?,” *IEEE Transactions on Medical Imaging*, vol. 35, no. 5, pp. 1299–1312, 2016, doi: 10.1109/TMI.2016.2535302.

[17] M. Raghu, C. Zhang, J. Kleinberg, and S. Bengio, “Transfusion: Understanding Transfer Learning for Medical Imaging,” in *Advances in Neural Information Processing Systems*, vol. 32, 2019, pp. 3342–3352.

[18] International Skin Imaging Collaboration, “ISIC 2018 Challenge—Task 1: Lesion Boundary Segmentation,” 2018. [Online]. Available: https://challenge.isic-archive.com/landing/2018/45/.

[19] J. Mongan, L. Moy, and C. E. Kahn, Jr., “Checklist for Artificial Intelligence in Medical Imaging (CLAIM): A Guide for Authors and Reviewers,” *Radiology: Artificial Intelligence*, vol. 2, no. 2, Art. e200029, 2020, doi: 10.1148/ryai.2020200029.

[20] D. Müller, I. Soto-Rey, and F. Kramer, “Towards a Guideline for Evaluation Metrics in Medical Image Segmentation,” *BMC Research Notes*, vol. 15, Art. 210, 2022, doi: 10.1186/s13104-022-06096-y.

[21] L. Maier-Hein *et al.*, “Why Rankings of Biomedical Image Analysis Competitions Should Be Interpreted with Care,” *Nature Communications*, vol. 9, Art. 5217, 2018, doi: 10.1038/s41467-018-07619-7.

[22] M. Wiesenfarth *et al.*, “Methods and Open-Source Toolkit for Analyzing and Visualizing Challenge Results,” *Scientific Reports*, vol. 11, Art. 2369, 2021, doi: 10.1038/s41598-021-82017-6.

[23] M. Benčević *et al.*, “Understanding Skin Color Bias in Deep Learning-Based Skin Lesion Segmentation,” *Computer Methods and Programs in Biomedicine*, vol. 245, Art. 108044, 2024, doi: 10.1016/j.cmpb.2024.108044.

[24] International Commission on Illumination, *Measurement of Human Skin Colour*, CIE 256:2025, 2025, doi: 10.25039/TR.256.2025.

[25] M. S. Sommers, B. Beacham, R. Baker, and J. Fargo, “Intra- and Inter-Rater Reliability of Digital Image Analysis for Skin Color Measurement,” *Skin Research and Technology*, vol. 19, no. 4, pp. 484–491, 2013, doi: 10.1111/srt.12072.

[26] M. Arnold, A. Ghosh, S. Ameling, and G. Lacey, “Automatic Segmentation and Inpainting of Specular Highlights for Endoscopic Imaging,” *EURASIP Journal on Image and Video Processing*, vol. 2010, Art. 814319, 2010, doi: 10.1155/2010/814319.

[27] S. Del Bino and F. Bernerd, “Variations in Skin Colour and the Biological Consequences of Ultraviolet Radiation Exposure,” *British Journal of Dermatology*, vol. 169, suppl. 3, pp. 33–40, 2013, doi: 10.1111/bjd.12529.

[28] International Electrotechnical Commission, *Multimedia Systems and Equipment—Colour Measurement and Management—Part 2-1: Colour Management—Default RGB Colour Space—sRGB*, IEC 61966-2-1:1999, 1999.

[29] International Commission on Illumination, *Colorimetry*, 4th ed., CIE 015:2018, 2018, doi: 10.25039/TR.015.2018.

[30] M. Banala, A. Mamidipaka, and T. Ogunleye, “Skin-Lightening Product Use Among South Asian Americans: Cross-Sectional Survey Study,” *JMIR Dermatology*, vol. 6, Art. e49068, 2023, doi: 10.2196/49068.

[31] J. E. Alderman *et al.*, “Tackling Algorithmic Bias and Promoting Transparency in Health Datasets: The STANDING Together Consensus Recommendations,” *The Lancet Digital Health*, vol. 7, no. 1, pp. e64–e88, 2025, doi: 10.1016/S2589-7500(24)00224-3.

[32] G. Sharma, W. Wu, and E. N. Dalal, “The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations,” *Color Research & Application*, vol. 30, no. 1, pp. 21–30, 2005, doi: 10.1002/col.20070.

[33] V. R. Weir *et al.*, “Evaluating Skin Tone Scales for Dermatologic Dataset Labeling: A Prospective-Comparative Study,” *npj Digital Medicine*, vol. 8, Art. 787, 2025, doi: 10.1038/s41746-025-02245-2.

[34] J. Mongan, L. Moy, S. H. Park, and C. E. Kahn, Jr., “Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update,” *Radiology: Artificial Intelligence*, vol. 6, no. 4, Art. e240300, 2024, doi: 10.1148/ryai.240300.

[35] J. M. Bland and D. G. Altman, “Statistical Methods for Assessing Agreement Between Two Methods of Clinical Measurement,” *The Lancet*, vol. 327, no. 8476, pp. 307–310, 1986.

[36] L. I.-K. Lin, “A Concordance Correlation Coefficient to Evaluate Reproducibility,” *Biometrics*, vol. 45, no. 1, pp. 255–268, 1989, doi: 10.2307/2532051.

[37] J. Cohen, “Weighted Kappa: Nominal Scale Agreement with Provision for Scaled Disagreement or Partial Credit,” *Psychological Bulletin*, vol. 70, no. 4, pp. 213–220, 1968, doi: 10.1037/h0026256.

## Fuentes internas de resultados

- `artifacts/yolov7/roi_margin.json`
- `artifacts/yolov7/validation_rois.json`
- `artifacts/selection/top3.json`
- `artifacts/color/lesion_margin.json`
- `results/test_original/*/results.json`
- `results/test_original/robustness_ita.json`
- `artifacts/test/mst/manifest.json`
- `artifacts/test/mst_rois.json`
- `results/test_mst/*/results.json`
- `results/test_mst/analysis.json`
- `artifacts/mskcc/census.json`
- `artifacts/mskcc/rois.json`
- `results/mskcc_analysis.json`
- `artifacts/final/provenance.json`
- Sección 19 de `10_METHODOLOGY_V2_FROM_ZERO.md`.
