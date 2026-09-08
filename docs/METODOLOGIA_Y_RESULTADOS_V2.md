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

### Cómo visualizar las fórmulas

Las ecuaciones usan el formato matemático de GitHub/MathJax: un signo `$` para matemática dentro de una línea y `$$` para ecuaciones centradas. Si el visor Markdown no implementa matemática, la siguiente tabla conserva las expresiones principales como texto plano, por lo que ningún valor o definición depende del renderizado LaTeX.

| Magnitud | Fórmula portable |
|---|---|
| Jaccard | `J = TP / (TP + FP + FN)` |
| Dice | `Dice = 2TP / (2TP + FP + FN)` |
| Jaccard umbralizado | `TJ = 0 si J < 0.65; en otro caso TJ = J` |
| Margen ROI congelado | `m* = Q95(max(rL, rR, rT, rB))` |
| Soporte limpio mínimo | `Nclean >= max(256, ceil(0.005 × área))` |
| Cambio MST pareado | `ΔJ(i,k) = J(i,k) - J(i,ORIGINAL)` |
| ITA | `ITA = (180/pi) × atan((L* - 50) / b*)` |
| Error absoluto medio | `MAE = mean(abs(ITApipe - ITAref))` |
| Error cuadrático medio | `RMSE = sqrt(mean((ITApipe - ITAref)^2))` |

## Metodología paso a paso

### Paso 1. Partición de datos y prevención de leakage

Se utilizó ISIC 2018 Task 1 [1], [18]. Las 2594 imágenes de `Training` se dividieron en cinco folds disjuntos. Cada imagen fue predicha por un detector que no la había visto durante entrenamiento, generando predicciones *out-of-fold* (OOF). Las 100 imágenes de `Validation` se reservaron para evaluar los 16 segmentadores y elegir el TOP-3. Las 1000 imágenes de `Test` se mantuvieron selladas hasta completar el `scientific freeze` [19], [34].

La caja de referencia de una máscara binaria de lesión $M_{GT}$ fue:

$$
B_{GT}=(x_{\min},y_{\min},x_{\max},y_{\max}),
$$

$$
x_{\min}=\min\{x:M_{GT}(x,y)=1\},\qquad
x_{\max}=\max\{x:M_{GT}(x,y)=1\},
$$

con una definición análoga para $y$. Los identificadores relacionados se mantuvieron agrupados cuando los metadatos lo permitieron; no se inventaron identidades faltantes.

### Paso 2. Entrenamiento del detector YOLOv7

YOLOv7 se eligió como localizador de lesión por su diseño *one-stage*, implementación pública y evidencia comparativa en dermatoscopia [2], [3]. Se utilizó una sola clase, `lesion`, entrada 640×640 con conservación de aspecto, checkpoint oficial `yolov7_training.pt`, ajuste completo de `backbone`, `neck` y `head`, minibatch físico 32 y acumulación de dos minibatches, es decir:

$$
B_{efectivo}=32\times 2=64.
$$

El optimizador fue SGD con Nesterov, $lr_0=0.005$, momentum 0.900, `weight_decay=0.0005`, *warm-up* de tres epochs y calendario cosenoidal One-Cycle hasta $lr_f=0.0005$. Cada fold se entrenó 300 epochs sin *early stopping*. El checkpoint se eligió con la *fitness* oficial [13]:

$$
F=0.1\,mAP@0.5+0.9\,mAP@0.5{:}0.95.
$$

Las augmentations, `ComputeLossOTA`, AutoAnchor y demás hiperparámetros siguieron la configuración oficial congelada. En inferencia se usaron confianza 0.25 y NMS IoU 0.45. Si la ejecución era válida pero no había caja, se registró `valid_no_detection` y se utilizó la imagen completa; un fallo técnico no se convirtió en una no-detección.

### Paso 3. Calibración OOF de la ROI

Para cada caja predicha OOF $B_i^p$, de ancho $w_i$ y alto $h_i$, se midió el déficit relativo necesario para cubrir la caja GT:

$$
r_{L,i}=\max\left(0,\frac{x^p_{\min}-x^{GT}_{\min}}{w_i}\right),\quad
r_{R,i}=\max\left(0,\frac{x^{GT}_{\max}-x^p_{\max}}{w_i}\right),
$$

$$
r_{T,i}=\max\left(0,\frac{y^p_{\min}-y^{GT}_{\min}}{h_i}\right),\quad
r_{B,i}=\max\left(0,\frac{y^{GT}_{\max}-y^p_{\max}}{h_i}\right).
$$

El margen individual y el margen global fueron:

$$
r_i=\max(r_{L,i},r_{R,i},r_{T,i},r_{B,i}),\qquad
m^*=Q_{0.95}(r_1,\ldots,r_N).
$$

También se cuantificó la fracción de lesión contenida por la ROI:

$$
C_i(m)=\frac{|M_i^{GT}\cap ROI_i(m)|}{|M_i^{GT}|}.
$$

Las no-detecciones se excluyeron de la estimación del percentil y se informaron por separado [4], [5].

### Paso 4. Inferencia de los 16 segmentadores en Validation

Los candidatos fueron 15 checkpoints conservados más GrabCut. Todos recibieron el mismo `roi_rgb` original, sin FOV, eliminación de pelo, *inpainting* ni transformación de color. Cada adaptador mantuvo su tamaño, normalización y regla de binarización nativos. La salida binaria de ROI se restauró a coordenadas de imagen completa y no recibió posprocesamiento morfológico común. Las máscaras vacías o llenas fueron predicciones científicas degeneradas; los fallos técnicos se corrigieron y reejecutaron [19], [20].

El Jaccard continuo y Dice fueron:

$$
J=\frac{TP}{TP+FP+FN},\qquad
Dice=\frac{2TP}{2TP+FP+FN}.
$$

La métrica primaria oficial de ISIC 2018 fue el Jaccard umbralizado [1], [18]:

$$
TJ_i=\begin{cases}
0,&J_i<0.65,\\
J_i,&J_i\ge 0.65,
\end{cases}
\qquad
\overline{TJ}=\frac{1}{N}\sum_{i=1}^{N}TJ_i.
$$

Se conservaron Jaccard, Dice, sensibilidad, especificidad, precisión, Boundary F1 y HD95 normalizado como métricas secundarias.

### Paso 5. Ranking, incertidumbre y TOP-3

Solo fueron elegibles los métodos con cobertura completa de las 100 imágenes de `Validation`. En cada una de 10 000 réplicas bootstrap se remuestrearon imágenes completas con reemplazo y se utilizó el mismo vector de índices para todos los métodos. Para una métrica $g$, la réplica $b$ fue:

$$
\hat g^{(b)}=\frac{1}{N}\sum_{j=1}^{N}g_{I_j^{(b)}},
\qquad I_j^{(b)}\sim\text{Uniforme}\{1,\ldots,N\}.
$$

Los percentiles 2.5 y 97.5 de las 10 000 réplicas formaron el IC95%. El orden puntual siguió $\overline{TJ}$; los desempates usaron Jaccard, Dice, Boundary F1 y menor HD95, en ese orden. Las distribuciones de posiciones se informaron porque los rankings biomédicos pueden depender fuertemente de los casos y de la métrica [21], [22].

### Paso 6. Construcción de `clean_skin_mask`

Cada método del TOP-3 produjo su propia máscara de piel limpia. No se fusionaron segmentadores. En la ROI, la piel candidata fue:

$$
M_{clean,m}=M_{ROI}\land \neg dilate(M_{lesion,m},r^*)
\land \neg M_{hair}\land \neg M_{highlight}.
$$

El detector morfológico de pelo se utilizó solo como exclusión colorimétrica, nunca para modificar la entrada del segmentador. Esta elección evita tratar un píxel sintetizado mediante *inpainting* como una medición real, aunque reconoce las limitaciones de los detectores morfológicos frente a pelo claro y falsos positivos [6]–[12]. Una cobertura de pelo mayor que 0.18 se conservó y marcó como alta.

El margen de lesión común se calibró como el Q95 de la distancia unilateral normalizada por el lado menor de la ROI. Las máscaras exactamente vacías o llenas produjeron colorimetría no disponible. Además, se exigió:

$$
N_{clean}\ge \max\left(256,\left\lceil0.005\,H_{ROI}W_{ROI}\right\rceil\right).
$$

Los reflejos saturados se excluyeron cuando:

$$
\max(R,G,B)\ge248
\quad\land\quad
\max(R,G,B)-\min(R,G,B)\le22.
$$

No se amplió el soporte, no se imputaron píxeles y no se revisaron imágenes manualmente.

### Paso 7. Conversión colorimétrica e ITA

Los JPEG se interpretaron como sRGB IEC 61966-2-1 [28]. Para cada canal normalizado $c_s\in[0,1]$, se aplicó la linealización:

$$
c=\begin{cases}
c_s/12.92,&c_s\le0.04045,\\
\left(\frac{c_s+0.055}{1.055}\right)^{2.4},&c_s>0.04045.
\end{cases}
$$

El RGB lineal se convirtió a XYZ D65 en `float64`:

$$
\begin{bmatrix}X\\Y\\Z\end{bmatrix}
=100
\begin{bmatrix}
0.4124564&0.3575761&0.1804375\\
0.2126729&0.7151522&0.0721750\\
0.0193339&0.1191920&0.9503041
\end{bmatrix}
\begin{bmatrix}R\\G\\B\end{bmatrix}.
$$

Con el blanco D65, $f(t)=t^{1/3}$ si $t>(6/29)^3$, y $f(t)=t/[3(6/29)^2]+4/29$ en otro caso, se obtuvo CIELAB [24], [29]:

$$
L^*=116f(Y/Y_n)-16,\quad
a^*=500[f(X/X_n)-f(Y/Y_n)],
$$

$$
b^*=200[f(Y/Y_n)-f(Z/Z_n)].
$$

El Individual Typology Angle se mantuvo como variable continua:

$$
ITA=\frac{180}{\pi}\operatorname{atan2}(L^*-50,b^*).
$$

No se transformó automáticamente ITA en Fitzpatrick ni MST. Se conservaron también mediana RGB, media RGB recortada 10–90 y mediana CIELAB [25], [27].

### Paso 8. `Scientific freeze` y apertura única de Test

Antes de materializar Test se congelaron commit, código, configuraciones, manifiestos, checkpoints, contenedor, dependencias, TOP-3, márgenes, semillas, métricas, paleta y análisis en `scientific_freeze.json` y un tag Git anotado. El verificador exigió repositorio limpio/publicado y ausencia de uso previo de Test. Después del freeze solo se permitieron reanudaciones técnicas con los mismos hashes [31], [34].

La apertura ocurrió una sola vez en CEDIA. Un DAG Slurm con dependencias `afterok` ejecutó inferencia ORIGINAL, métricas, síntesis MST, YOLOv7 por condición, TOP-3 por condición y análisis. La GT se utilizó únicamente en evaluación y en el soporte autorizado de síntesis; nunca se entregó a YOLOv7 ni a los segmentadores.

### Paso 9. Evaluación del Test original

YOLOv7 se ejecutó sobre las 1000 imágenes completas. Cada TOP-3 recibió la ROI dinámica o la imagen completa si correspondía `valid_no_detection`. Las máscaras se restauraron a dimensiones originales y se compararon con la GT. Además se estimó un ITA independiente de los segmentadores usando GT dilatada, pelo y reflejos, y se calculó Spearman entre ITA y Jaccard, Dice y Boundary F1. Esta relación se interpretó como robustez frente al ITA observado, no como fairness demográfica [23], [31].

### Paso 10. Síntesis MST de imagen completa

La paleta MST 01–10 se definió en un único artefacto sRGB protegido por hash. Para cada imagen original se estimó una mediana Lab sobre soporte independiente de los segmentadores:

$$
M_{support}=M_{imagen}\land\neg dilate(M_{GT},r^*)
\land\neg M_{hair}\land\neg M_{highlight}.
$$

Se exigió el mismo mínimo de soporte, ahora respecto del área completa. Si no se cumplía, las diez condiciones se registraban `unavailable` sin lanzar excepción. Para el tono $k$:

$$
\Delta Lab_k=Lab_{target,k}-Lab_{source},
$$

$$
Lab'_k(x,y)=Lab_{original}(x,y)+\Delta Lab_k
\quad\forall(x,y)\text{ de la imagen completa}.
$$

Después se convirtió nuevamente a sRGB, se aplicó *clipping* al dominio válido y se guardó PNG RGB de 8 bits, sin pérdida, con `compress_level=6`. Se verificaron dimensiones, GT idéntica, ausencia de NaN/Inf, hashes de píxeles decodificados y correspondencia fuente–condición. CIEDE2000 se calculó según Sharma, Wu y Dalal [32].

### Paso 11. Evaluación end-to-end de cada condición MST

YOLOv7 se volvió a ejecutar independientemente en cada imagen MST completa. Cada segmentador recibió solo la ROI propia de esa condición, o la imagen completa ante no-detección válida. La variable primaria fue:

$$
\Delta J_{i,m,k}=J_{i,m,k}-J_{i,m,ORIGINAL}.
$$

Cada tono se mantuvo separado. Se calcularon media, mediana, IQR e IC95% por bootstrap pareado, además de la peor caída por fuente:

$$
\Delta J^{worst}_{i,m}=\min_{k\in\{1,\ldots,10\}}\Delta J_{i,m,k}.
$$

No se hicieron pruebas de hipótesis ni valores $p$ para MST, porque las condiciones son medidas repetidas sintéticas y no observaciones demográficas independientes.

### Paso 12. Censo y procesamiento MSKCC

Se censaron las 4879 imágenes públicas de MSKCC y se conservaron todas las tomas disponibles. La jerarquía fue `patient_id → sitio/lesión → modo de adquisición → imagen`. Los valores de referencia ausentes quedaron `NA`; no se imputaron ni se seleccionaron casos según las salidas. MSKCC se usó solo para concordancia cromática, porque no existe una GT de lesión autorizada para validar segmentación [33], [34].

YOLOv7 produjo una ROI dinámica por imagen. Si no detectó, se aplicó el mismo fallback automático a imagen completa. Cada TOP-3 generó su máscara de lesión, `clean_skin_mask` y variables cromáticas de forma independiente.

### Paso 13. Concordancia MSKCC

La medida primaria fue ICC de acuerdo absoluto entre el ITA del pipeline y el ITA del colorímetro, con bootstrap por paciente. En forma general, para un modelo de dos vías de acuerdo absoluto:

$$
ICC(A,1)=\frac{MS_R-MS_E}
{MS_R+(k-1)MS_E+\frac{k}{n}(MS_C-MS_E)},
$$

donde $MS_R$, $MS_C$ y $MS_E$ son cuadrados medios de filas, columnas y error; $n$ es el número de unidades y $k$ el número de mediciones. Cada réplica remuestreó pacientes completos.

Los errores secundarios fueron:

$$
Bias=\frac{1}{N}\sum_i(ITA_{pipe,i}-ITA_{ref,i}),
$$

$$
MAE=\frac{1}{N}\sum_i|ITA_{pipe,i}-ITA_{ref,i}|,
\quad
RMSE=\sqrt{\frac{1}{N}\sum_i(ITA_{pipe,i}-ITA_{ref,i})^2}.
$$

Los límites de Bland–Altman fueron [35]:

$$
LoA=Bias\pm1.96\,SD(ITA_{pipe}-ITA_{ref}).
$$

Para la referencia MST ordinal se usaron Kendall $\tau_b$ y Spearman $\rho$, sin convertir ITA a una categoría MST [33], [36], [37].

### Paso 14. Reanudación, hashes y cierre

Las etapas publicaron manifiestos atómicamente. Las inferencias MST mantuvieron un `progress.jsonl` por método y solo reutilizaron una máscara si coincidían método, ROI, procedencia y SHA-256. Las variantes PNG existentes se recalcularon y reutilizaron únicamente si coincidían dimensiones y hash de píxeles decodificados. Los fallos quedaron en la bitácora; no se borraron checkpoints, logs ni ledger. El job `25037` verificó los gates detector, selección, freeze, MST, MSKCC y procedencia y emitió `V2_COMPLETE`.

## Resultados reales

### Valores finales retenidos en cada etapa

Esta tabla responde explícitamente qué valor se adoptó —y no solo qué se calculó— antes de pasar a la etapa siguiente. Los valores congelados no se reajustaron después de abrir Test.

| Etapa | Valor finalmente retenido | Uso posterior |
|---|---|---|
| Partición | Training = 2594; Validation = 100; Test = 1000 | Desarrollo OOF, selección y evaluación final, respectivamente |
| Detector | YOLOv7, confianza 0.25, NMS IoU 0.45, entrada 640, batch efectivo 64 | Localización uniforme en Validation, Test, MST y MSKCC |
| Margen dinámico de ROI (D09) | $m^*=0.1225400188$ (12.254% del ancho/alto de la caja) | Expandir cada caja detectada; imagen completa si no hay detección válida |
| Candidatos de segmentación | 15 checkpoints preservados + GrabCut = 16 métodos | Evaluación completa sobre Validation |
| TOP-3 congelado | AViT; DeLightSAM-Dermoscopy; VM-UNet ISIC17 | Únicos métodos evaluados en Test, MST y MSKCC |
| Margen de exclusión de lesión (D35) | $r^*=0.1418421924$ del lado menor de la ROI | Dilatar por método su máscara antes de medir piel limpia |
| Soporte colorimétrico (D37/D59) | $\max(256,\lceil0.005\,A\rceil)$ píxeles válidos | Menos soporte implica `unavailable`; nunca imputación |
| Umbral de pelo alto | cobertura > 0.18 | Marca de calidad; no elimina la observación por sí sola |
| Reflejo saturado | máximo RGB ≥ 248 y rango RGB ≤ 22 | Exclusión del soporte colorimétrico |
| Scientific freeze | commit `1c6ddbe8838d0734e08cb5eaa5d009219212f28c`; semilla 20260828 | Autorizó la única apertura de Test |
| Test original | 983 detecciones; 17 `valid_no_detection` | Las 17 usaron imagen completa |
| Universo MST | 975 fuentes válidas; 25 no disponibles | 9750 variantes y 250 condiciones indisponibles |
| Codificación MST | imagen completa, PNG sin pérdida, `compress_level=6` | Evita el parche rectangular y reduce tamaño sin cambiar píxeles |
| Análisis MSKCC | censo completo de 4879 imágenes | Concordancia de ITA, no exactitud de segmentación |

### Tabla general del pipeline

| Etapa | Universo/resultado | Evidencia principal |
|---|---:|---|
| ISIC Training | 2594 imágenes; 5 folds OOF | `roi_margin.json` |
| OOF YOLOv7 | 2551 detecciones válidas; 43 no-detecciones | margen ROI $m^*=0.122540$ |
| Validation | 100 imágenes; 16 métodos completos | `top3.json` |
| TOP-3 V2 | AViT, DeLightSAM-Dermoscopy, VM-UNet ISIC17 | selección D28–D32 |
| Margen de piel D35 | 300 observaciones método–imagen | $r^*=0.141842$ |
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

### Calibración OOF del margen de ROI

El valor retenido fue $m^*=0.1225400188$. Se estimó exclusivamente con las 2551 detecciones OOF válidas; las 43 no-detecciones se informaron pero no entraron en el percentil.

| Fold OOF | Imágenes | Detectadas | `valid_no_detection` | Mediana del margen mínimo individual |
|---:|---:|---:|---:|---:|
| 0 | 519 | 511 | 8 | 0.011719 |
| 1 | 519 | 507 | 12 | 0.011362 |
| 2 | 519 | 511 | 8 | 0.011734 |
| 3 | 519 | 513 | 6 | 0.009328 |
| 4 | 518 | 509 | 9 | 0.012522 |
| **Total** | **2594** | **2551** | **43** | **$m^*=Q_{0.95}=0.122540$** |

En Validation, el margen congelado produjo 95 detecciones y cinco `valid_no_detection`. No se recalibró con esas 100 imágenes.

### Ranking completo de los 16 métodos en Validation

Todas las medias corresponden a $N=100$. `P(TOP-3)` es la proporción de réplicas bootstrap en que el método ocupó una de las tres primeras posiciones.

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

### Colorimetría en Validation y margen D35

El margen finalmente congelado fue $r^*=0.1418421924$. Se obtuvo de 300 observaciones (100 por cada método TOP-3), aplicando primero el Q95 de distancia unilateral por par método–imagen y después el Q95 global.

| Método | N para D35 | Media de distancia Q95 normalizada | Mediana | Mínimo | Máximo |
|---|---:|---:|---:|---:|---:|
| AViT | 100 | 0.033568 | 0.011699 | 0.000000 | 0.581813 |
| DeLightSAM-Dermoscopy | 100 | 0.032283 | 0.009861 | 0.000000 | 0.390625 |
| VM-UNet ISIC17 | 100 | 0.040222 | 0.015740 | 0.000000 | 0.341411 |
| **Regla común retenida** | **300** | — | — | — | **$r^*=0.141842$** |

| Método | Color disponible | No disponible | Pelo alto | Cobertura limpia media disponible |
|---|---:|---:|---:|---:|
| AViT | 100 | 0 | 0 | 0.183971 |
| DeLightSAM-Dermoscopy | 100 | 0 | 0 | 0.187691 |
| VM-UNet ISIC17 | 99 | 1 | 0 | 0.204212 |

La única ausencia fue `unavailable_insufficient_clean_skin` en VM-UNet ISIC17. Se mantuvo como no disponible y no se sustituyó por otra máscara.

### TOP-3 en Test original — métricas completas

| Método | N | TJ | Jaccard | Dice | Sensibilidad | Especificidad | Precisión | Exactitud | Boundary F1 | HD95 norm. | Tiempo medio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AViT | 1000 | 0.766669 | 0.809185 | 0.880602 | 0.938563 | 0.932968 | 0.858766 | 0.935530 | 0.498894 | 0.062209 | 2.573 s |
| DeLightSAM-Dermoscopy | 1000 | 0.742512 | 0.791406 | 0.869111 | 0.923753 | 0.936752 | 0.853147 | 0.929580 | 0.435713 | 0.062970 | 2.368 s |
| VM-UNet ISIC17 | 1000 | 0.754465 | 0.798340 | 0.869385 | 0.907274 | 0.947366 | 0.874426 | 0.929293 | 0.529279 | 0.060656 | 4.264 s |

El análisis de robustez frente al ITA continuo tuvo $N=966$ pares disponibles por método:

| Método | $\rho$ ITA–Jaccard (IC95%) | $\rho$ ITA–Dice (IC95%) | $\rho$ ITA–Boundary F1 (IC95%) |
|---|---:|---:|---:|
| AViT | 0.0479 [−0.0178, 0.1128] | 0.0479 [−0.0178, 0.1128] | 0.0621 [−0.0016, 0.1281] |
| DeLightSAM-Dermoscopy | 0.0022 [−0.0642, 0.0693] | 0.0022 [−0.0642, 0.0693] | 0.0004 [−0.0636, 0.0633] |
| VM-UNet ISIC17 | 0.1154 [0.0502, 0.1794] | 0.1154 [0.0502, 0.1794] | 0.1248 [0.0603, 0.1879] |

### Sensibilidad MST end-to-end

Cada celda de método muestra la media de $\Delta J$ y su IC95% bootstrap. Todas las condiciones tienen $N=975$ fuentes disponibles. La tasa de detección corresponde a YOLOv7 y es común a los tres segmentadores porque el detector se ejecutó una vez por condición completa.

| Condición | Detección YOLOv7 | AViT $\Delta J$ | DeLightSAM $\Delta J$ | VM-UNet ISIC17 $\Delta J$ |
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

#### Detector YOLOv7 por condición MST

Cada fila contiene las 975 fuentes válidas. `N caja` es el número de detecciones usado para resumir las métricas geométricas; las restantes son `valid_no_detection` y pasan la imagen completa al segmentador.

| Condición | N | Tasa detección | N caja | Contención media de caja GT | Contención media de lesión | Inflación media de ROI |
|---|---:|---:|---:|---:|---:|---:|
| MST 01 | 975 | 0.964103 | 940 | 0.604255 | 0.961709 | 2.156717 |
| MST 02 | 975 | 0.972308 | 948 | 0.603376 | 0.964221 | 2.110710 |
| MST 03 | 975 | 0.978462 | 954 | 0.593291 | 0.968598 | 2.100615 |
| MST 04 | 975 | 0.981538 | 957 | 0.584117 | 0.969107 | 1.911922 |
| MST 05 | 975 | 0.979487 | 955 | 0.557068 | 0.967432 | 1.872984 |
| MST 06 | 975 | 0.984615 | 960 | 0.563542 | 0.966527 | 1.761722 |
| MST 07 | 975 | 0.989744 | 965 | 0.583420 | 0.972613 | 1.780593 |
| MST 08 | 975 | 0.989744 | 965 | 0.624870 | 0.981345 | 1.922265 |
| MST 09 | 975 | 0.995897 | 971 | 0.613800 | 0.980822 | 1.935901 |
| MST 10 | 975 | 0.998974 | 974 | 0.644764 | 0.986669 | 2.049716 |

#### Detalle MST — AViT

| Condición | N | ΔJ media [IC95%] | ΔDice | ΔBoundary F1 | ΔTJ |
|---|---:|---:|---:|---:|---:|
| MST 01 | 975 | −0.022882 [−0.031531, −0.014775] | −0.019648 | −0.052024 | −0.031826 |
| MST 02 | 975 | −0.017845 [−0.025727, −0.010333] | −0.015594 | −0.037939 | −0.023931 |
| MST 03 | 975 | −0.014838 [−0.022340, −0.007350] | −0.011546 | −0.042188 | −0.019985 |
| MST 04 | 975 | −0.012481 [−0.019438, −0.005825] | −0.010079 | −0.035986 | −0.019098 |
| MST 05 | 975 | −0.013679 [−0.019955, −0.007690] | −0.011028 | −0.035316 | −0.021365 |
| MST 06 | 975 | −0.007745 [−0.013856, −0.001877] | −0.006884 | −0.023159 | −0.006248 |
| MST 07 | 975 | −0.006698 [−0.012819, −0.000528] | −0.004394 | −0.034398 | −0.007994 |
| MST 08 | 975 | −0.014831 [−0.021029, −0.008348] | −0.007247 | −0.086698 | −0.025813 |
| MST 09 | 975 | −0.025631 [−0.032665, −0.018441] | −0.015823 | −0.119767 | −0.039558 |
| MST 10 | 975 | −0.026562 [−0.034581, −0.018226] | −0.014700 | −0.126327 | −0.044903 |

#### Detalle MST — DeLightSAM-Dermoscopy

| Condición | N | ΔJ media [IC95%] | ΔDice | ΔBoundary F1 | ΔTJ |
|---|---:|---:|---:|---:|---:|
| MST 01 | 975 | −0.033805 [−0.043147, −0.024939] | −0.030103 | −0.063705 | −0.042266 |
| MST 02 | 975 | −0.030490 [−0.038936, −0.022270] | −0.026347 | −0.065257 | −0.038430 |
| MST 03 | 975 | −0.031022 [−0.039107, −0.023163] | −0.025329 | −0.070856 | −0.041648 |
| MST 04 | 975 | −0.022234 [−0.029655, −0.014978] | −0.017589 | −0.049367 | −0.031574 |
| MST 05 | 975 | −0.014885 [−0.021772, −0.008351] | −0.013292 | −0.032496 | −0.011807 |
| MST 06 | 975 | −0.005955 [−0.012322, 0.000110] | −0.006653 | 0.004652 | 0.000115 |
| MST 07 | 975 | −0.000645 [−0.006977, 0.005573] | −0.001370 | 0.018466 | 0.001717 |
| MST 08 | 975 | 0.005681 [−0.000563, 0.012252] | 0.004991 | 0.034300 | 0.005050 |
| MST 09 | 975 | 0.005619 [−0.001895, 0.013157] | 0.003283 | 0.056487 | 0.004203 |
| MST 10 | 975 | 0.003739 [−0.005161, 0.012335] | 0.002948 | 0.060826 | 0.000381 |

#### Detalle MST — VM-UNet ISIC17

| Condición | N | ΔJ media [IC95%] | ΔDice | ΔBoundary F1 | ΔTJ |
|---|---:|---:|---:|---:|---:|
| MST 01 | 975 | −0.040355 [−0.050752, −0.029927] | −0.035970 | −0.055639 | −0.057704 |
| MST 02 | 975 | −0.039681 [−0.049614, −0.030104] | −0.034641 | −0.062296 | −0.054891 |
| MST 03 | 975 | −0.039563 [−0.049376, −0.029984] | −0.032498 | −0.073543 | −0.055502 |
| MST 04 | 975 | −0.037811 [−0.047498, −0.028610] | −0.030898 | −0.075903 | −0.051744 |
| MST 05 | 975 | −0.040447 [−0.049989, −0.031316] | −0.033035 | −0.077094 | −0.057564 |
| MST 06 | 975 | −0.148655 [−0.163792, −0.134061] | −0.122562 | −0.167150 | −0.224608 |
| MST 07 | 975 | −0.137717 [−0.153115, −0.122971] | −0.112267 | −0.162476 | −0.214829 |
| MST 08 | 975 | −0.151778 [−0.169755, −0.134439] | −0.131652 | −0.157149 | −0.213033 |
| MST 09 | 975 | −0.260425 [−0.281997, −0.239121] | −0.239934 | −0.234469 | −0.346870 |
| MST 10 | 975 | −0.340518 [−0.364386, −0.317536] | −0.323244 | −0.269811 | −0.437189 |

En estas tablas, el valor conservado para interpretar robustez es el cambio pareado respecto de la misma fuente ORIGINAL. No se comparan grupos distintos: cada una de las 975 imágenes contribuye con su propio control.

### Concordancia cromática MSKCC

Los denominadores difieren porque D36–D37 conservan como `NA` los casos sin soporte válido y porque no todas las imágenes tienen la referencia requerida. Los IC95% del ICC remuestrean pacientes completos.

| Flujo MSKCC | Imágenes |
|---|---:|
| Censo íntegro | 4879 |
| YOLOv7 detectó lesión | 2825 |
| `valid_no_detection`, imagen completa | 2054 |

| Método | N continuo | Pacientes | ICC absoluto (IC95%) | Bias ITA | MAE | RMSE | Bland–Altman LoA |
|---|---:|---:|---:|---:|---:|---:|---:|
| AViT | 1168 | 46 | 0.3357 [0.2367, 0.4247] | −4.0891 | 60.2784 | 75.7678 | [−152.4409, 144.2628] |
| DeLightSAM-Dermoscopy | 699 | 46 | 0.4490 [0.3590, 0.5216] | 14.7545 | 47.9649 | 61.3627 | [−102.0714, 131.5804] |
| VM-UNet ISIC17 | 1396 | 46 | 0.5145 [0.4146, 0.5894] | 11.5159 | 47.6528 | 59.8357 | [−103.6108, 126.6426] |

| Método | N con MST | Kendall $\tau_b$ | Spearman $\rho$ |
|---|---:|---:|---:|
| AViT | 3735 | −0.5500 | −0.6793 |
| DeLightSAM-Dermoscopy | 3092 | −0.6237 | −0.7786 |
| VM-UNet ISIC17 | 3873 | −0.6251 | −0.7770 |

### Integridad de los artefactos finales

| Artefacto | SHA-256 |
|---|---|
| `artifacts/selection/top3.json` | `0cb2513bd59c231b40ce8c909f523e9ad2695005b7b3e388512531b26b262ecf` |
| `results/color_validation/colorimetry.json` | `56eac5fa24dd13482d4ff2fac06112acd89f8450fa97dfbc1c2b49dc06e7292a` |
| `results/test_original/avit/results.json` | `dfb072b3471d9adc8790b278ebd7fdb11ebe081c6f8f522c2b652bb6a022d6ba` |
| `results/test_original/delightsam-dermoscopy/results.json` | `bbe6923f531fbad2ecf675e892998a877976cfc3a4f89c24db3e5a885ffbcf77` |
| `results/test_original/vmunet-isic17/results.json` | `8cb6462c9a5a97fef073e4c86eb12252f7dc9a3ecd83a06b5051a7feeebd5e1b` |
| `results/test_mst/analysis.json` | `9ec78ea56d2bc7e4315cc6fc70a9584fc7096c7173233175c208329a2fe59786` |
| `artifacts/mskcc/census.json` | `d2c1b142755f8debd856d26dd40c95d82e32e813b0f314d5647c4f914bd9fa6a` |
| `artifacts/mskcc/rois.json` | `d95fdb9bb331f7cadc82d501790bc48cda9b5c85222f1ec149c5eb5ef1a2fe17` |
| `results/mskcc_analysis.json` | `4d3559ecaa93655f9298094bd6e2ddfc1cb9e8c0044842677f5e9def4177e542` |
| `artifacts/final/provenance.json` | `581c1f26126477104e91dce592b2a45b093bd2bfcf93a87ca81feb2eb46872e5` |

## Auditoría exhaustiva del canon y de la ejecución

Esta sección complementa el recorrido narrativo anterior con matrices de cobertura. Distingue: **canon** (D01–D60 y `methodology_v2.json`), **implementación** (código versionado), **evidencia** (JSON finales de CEDIA) y **valor retenido** (parámetro que pasó al siguiente gate). Que una referencia justifique una familia de métodos no significa que prescriba el valor numérico adoptado: esos valores son decisiones explícitas del canon.

### Matriz completa D01–D60

| Decisión | Bloque | Regla normativa resumida |
|---|---|---|
| D01 | datos/gobernanza | Dataset principal: **ISIC 2018 Task 1 — Lesion Boundary Segmentation** |
| D02 | datos/gobernanza | Detector de ROI: **YOLOv7** |
| D03 | datos/gobernanza | Desarrollo interno mediante **5-fold CV disjoint + predicciones OOF** sobre `Training` |
| D04 | datos/gobernanza | `Validation` oficial se conserva como conjunto independiente de desarrollo/selección |
| D05 | datos/gobernanza | `Test` oficial de ISIC 2018 permanece **sellado hasta el final** |
| D06 | datos/gobernanza | El antiguo subconjunto de **64 imágenes** no pertenece a V2 y no participa en selección ni evaluación final |
| D07 | datos/gobernanza | YOLOv7 recibe la **imagen dermatoscópica original** |
| D08 | datos/gobernanza | El procesamiento **FOV se elimina completamente** de V2 |
| D09 | YOLO/ROI | La ROI usa margen simétrico calibrado como **percentil 95 del margen mínimo requerido en predicciones OOF válidas de YOLOv7** |
| D10 | YOLO/ROI | **No** se realiza hair removal ni inpainting antes de la segmentación |
| D11 | YOLO/ROI | La `hair_mask` se conserva únicamente para **excluir píxeles contaminados del análisis posterior de piel**, no para modificar la entrada del segmentador |
| D12 | pelo | Para la `hair_mask` se conserva el **detector morfológico determinista ya implementado en el proyecto**, adaptado a V2 para operar sobre la ROI y devolver solo máscara; no se introduce un modelo deep-learning adicional |
| D13 | pelo | Si `hair_coverage_fraction > 0.18`, se **conserva la `hair_mask`**, se activa `hair_mask_high_coverage` y sus píxeles continúan excluidos del análisis de piel; no se sustituye por una máscara vacía ni se invalida automáticamente la imagen |
| D14 | pelo | Especificación de arquitectura: **YOLOv7**, sin seleccionar ni adoptar una variante con sufijo (`tiny`, `x`, `w6`, `e6`, `d6` o `e6e`) |
| D15 | pelo | Los cinco folds de YOLOv7 se inicializan con el checkpoint oficial de transferencia **`yolov7_training.pt`**; se fija el mismo archivo y SHA-256 para todos los folds, y los checkpoints propios solo pueden reanudar su fold de origen |
| D16 | pelo | YOLOv7 se ajusta de forma completa desde el primer epoch: **`backbone`, `neck` y `head` permanecen entrenables**, sin congelamiento inicial ni permanente |
| D17 | pelo | Entrada de YOLOv7 a resolución fija de **640 × 640 px**, conservando relación de aspecto mediante *padding*; no se activa `--multi-scale` |
| D18 | pelo | Minibatch físico de **32 imágenes** y acumulación de **2 minibatches**, para un batch efectivo de **64 imágenes por actualización** en todos los folds |
| D19 | entrenamiento YOLO | Jerarquía de configuración de YOLOv7: usar primero los valores explícitos de la mejor configuración reportada por AlSadhan *et al.* [3]; para parámetros omitidos o ambiguos, usar la implementación oficial [13], sin completar vacíos por inferencia |
| D20 | entrenamiento YOLO | Optimización YOLOv7: **SGD con Nesterov**, `lr0=0.005`, `momentum=0.900`, `weight_decay=0.0005`, 3 epochs de *warm-up* y calendario One-Cycle cosenoidal hasta `lrf=0.1` |
| D21 | entrenamiento YOLO | Duración y checkpoint de YOLOv7: **300 epochs fijos**, sin *early stopping*; cada fold congela `best.pt` mediante la *fitness* oficial sobre su validación interna y el reajuste final usa `Validation` oficial para seleccionar `best.pt` |
| D22 | entrenamiento YOLO | Augmentations de YOLOv7: adoptar exactamente el bloque de `hyp.scratch.custom.yaml` oficial para transferencia; aplicarlo *online* solo al subconjunto de entrenamiento, sin augmentation de validación/test ni dataset offline expandido |
| D23 | entrenamiento YOLO | Pérdida y anchors de YOLOv7: adoptar `ComputeLossOTA` y los valores oficiales de `hyp.scratch.custom.yaml`, sin focal loss ni *label smoothing*; mantener AutoAnchor activo y aislado al entrenamiento de cada fold |
| D24 | entrenamiento YOLO | Inferencia YOLOv7: fijar los valores oficiales `confidence_threshold=0.25` y `nms_iou_threshold=0.45`, sin calibración OOF ni herencia de los umbrales YOLOv3 |
| D25 | entrenamiento YOLO | No-detección YOLOv7: una inferencia válida sin cajas usa la **imagen completa de la condición evaluada** como ROI y registra `valid_no_detection`; esto aplica tanto a `ORIGINAL` como a cada variante MST, mientras cualquier fallo técnico detiene la ejecución |
| D26 | segmentación/ranking | Contrato de segmentadores: entrada común `roi_rgb` original y salida binaria en coordenadas ROI; cada checkpoint conserva su preprocesamiento y decisión nativos, con trazabilidad completa y separación entre predicción degenerada y fallo técnico |
| D27 | segmentación/ranking | Posprocesamiento de lesión: **ningún posprocesamiento morfológico común**; la máscara científica final es exactamente la salida binaria restaurada por D26, y los diagnósticos de calidad se registran sin alterarla ni excluirla |
| D28 | segmentación/ranking | Métrica primaria para ordenar segmentadores: **media macro del Jaccard umbralizado oficial de ISIC 2018 (`T=0.65`)**; se conservan todas las métricas históricas aplicables como resultados secundarios, sin reutilizar el puntaje compuesto de 100 puntos |
| D29 | segmentación/ranking | Incertidumbre del ranking: **10 000 remuestreos bootstrap pareados por imagen**, con intervalos de métricas y posiciones; las comparaciones pareadas por permutación con ajuste de Holm son secundarias y no sustituyen el orden por Jaccard umbralizado |
| D30 | segmentación/ranking | Elegibilidad del ranking: **cobertura científica completa del conjunto de selección**; los fallos técnicos se corrigen y reejecutan bajo una revisión trazable o el método queda inelegible, sin imputar ceros ni mezclar revisiones |
| D31 | segmentación/ranking | Número de segmentadores seleccionados: **TOP-3** elegible según D28–D30; se conserva únicamente `k=3`, no las identidades históricas S01/S10/S14, y los tres candidatos continúan a validación y robustez |
| D32 | segmentación/ranking | Desempate TOP-3: usar, con precisión completa, **Jaccard umbralizado, Jaccard, Dice, Boundary F1 y menor HD95 normalizado**, en ese orden; si la igualdad persiste, detener y documentar una nueva decisión |
| D33 | colorimetría | Propagación hacia colorimetría: cada integrante del TOP-3 genera su propia **`clean_skin_mask_m`** y sus propias variables cromáticas; no se fusionan las tres máscaras ni se reduce el análisis al TOP-1 |
| D34 | colorimetría | Dominio de piel candidata: usar **todo el complemento disponible dentro de la ROI**, excluyendo la máscara de lesión dilatada y la `hair_mask`; no usar anillo exterior, FOV ni *fallback* espacial |
| D35 | colorimetría | Margen de seguridad de piel: calibrar una **distancia unilateral Q95 común al TOP-3** en el conjunto de selección, normalizada por el lado menor de la ROI y congelada antes de validación externa/test |
| D36 | colorimetría | Máscara de lesión degenerada: si está exactamente vacía o llena, la colorimetría de ese método–imagen queda **no disponible (`null/NA`)**, sin tomar otra máscara, GT ni toda la ROI como sustituto |
| D37 | colorimetría | Suficiencia de piel candidata: exigir **`max(256, ceil(0.005 × área_ROI))` píxeles**; si no se alcanza, la colorimetría queda `null/NA`, sin ampliar el dominio ni recuperar píxeles excluidos |
| D38 | colorimetría | Reflejos saturados: excluir en el RGB original los píxeles con **`max(R,G,B)≥248` y `max−min≤22`**, sin dilatación ni inpainting y con una máscara común para el TOP-3 |
| D39 | colorimetría | Otros artefactos: **no añadir revisión visual, máscara manual ni detector automático adicional**; `clean_skin_mask_m` se genera de extremo a extremo únicamente con ROI, lesión+margen, `hair_mask` y reflejos D38 |
| D40 | colorimetría | Variables cromáticas: conservar **mediana RGB, media RGB recortada 10–90, mediana CIELAB e ITA**, además de conteo/cobertura; ITA es principal pero se interpreta como medida relativa de la imagen |
| D41 | colorimetría | Conversión cromática: decodificar todo JPEG como **sRGB IEC 61966-2-1**, linealizar explícitamente y convertir en `float64` a XYZ D65 y CIELAB CIE; no usar perfiles ICC por imagen ni redondear valores intermedios |
| D42 | colorimetría | Interpretación de ITA: conservar **`ita_degrees` únicamente como variable continua**; no asignar categorías ITA ni convertirlo automáticamente a Fitzpatrick o Monk, y mantener separadas las etiquetas externas legítimas |
| D43 | MST | Función del experimento MST: análisis **secundario, pareado y sintético** sobre el Test sellado después del `scientific freeze`; no modifica el TOP-3 ni constituye validación clínica o prueba de equidad demográfica real |
| D44 | MST | Paleta MST: un único artefacto PNG sRGB con **diez franjas sólidas MST 01–10**, generado desde la misma configuración numérica usada por la síntesis, serializado con compresión PNG sin pérdida fija y protegido por hash |
| D45 | MST | Síntesis MST: estimar `delta_Lab` sobre la **imagen `ORIGINAL` completa** excluyendo GT dilatada, pelo y reflejos, y aplicar el mismo desplazamiento a **todos los píxeles de la imagen completa, incluida la lesión**; no usar máscaras predichas por los segmentadores ni entregar GT a YOLOv7 o a los segmentadores |
| D46 | MST | QC de síntesis MST: aplicar únicamente **invariantes estructurales automáticos**, incluida identidad de dimensiones con la imagen completa y serialización PNG RGB de 8 bits con compresión sin pérdida fija; registrar `delta_L*`, `delta_a*`, `delta_b*`, CIEDE2000 y clipping como variables continuas, sin revisión manual n |
| D47 | MST | Métrica primaria de sensibilidad MST: **cambio pareado del Jaccard continuo end-to-end respecto de `ORIGINAL`**, después de volver a ejecutar YOLOv7 y el segmentador en cada condición; el Jaccard umbralizado conserva la primacía del Test original y las métricas del detector y del segmentador permanecen como secundarias |
| D48 | MST | Resumen MST: conservar **los diez tonos por separado** y reportar media, mediana, IQR e IC95% con 10 000 bootstrap pareados por imagen, además de la peor caída por fuente, la disponibilidad y los cambios del detector por condición; no crear un score único ni reordenar el TOP-3 |
| D49 | MST | Inferencia MST: **no realizar pruebas de hipótesis ni reportar valores p**; presentar magnitudes e IC95% bootstrap para cambios por tono y diferencias pareadas entre segmentadores, sin afirmar fairness real |
| D50 | MSKCC | Cohorte MSKCC: utilizarla exclusivamente para evaluar **qué tan bien el pipeline completo recupera el tono de piel** frente a sus referencias MST/colorimétricas; no realizar validación externa del segmentador, no usar IMA++ y no modificar el TOP-3 |
| D51 | MSKCC | Censo MSKCC: procesar **todas las imágenes disponibles y todos los pacientes**, sin muestra objetivo ni límite por paciente; conservar como mediciones repetidas las distintas tomas de un mismo sitio y analizar los resultados por MST, posición anatómica y tipo de toma |
| D52 | MSKCC | Ejecución V2: desarrollar en el repositorio nuevo **`Bench_Fairness_V2`**, reutilizar de forma trazable solo los archivos necesarios del repositorio reparado, ejecutar todo resultado científico exclusivamente en CEDIA sobre `compute-0-2` excluyendo `compute-0-1`, versionar mediante commit/push, disponer de scripts por  |
| D53 | aprobaciones/ejecución | Concordancia MSKCC: usar como análisis primario el **ICC de acuerdo absoluto** entre `ita_degrees` del pipeline y el ITA del colorímetro, con IC95% por bootstrap de pacientes; reportar sesgo, MAE, RMSE y Bland–Altman como secundarios, desagregar por método/sitio/dispositivo/modo y evaluar MST solo mediante asociación o |
| D54 | aprobaciones/ejecución | Robustez real y límites de fairness: sobre ISIC 2018 Test `ORIGINAL`, después del freeze, relacionar el ITA continuo estimado independientemente de los modelos mediante soporte GT con Jaccard, Dice y Boundary F1 del TOP-3 usando Spearman y diferencias bootstrap; en MSKCC limitarse a errores cromáticos D53 desagregados; |
| D55 | aprobaciones/ejecución | `Scientific freeze`: congelar antes de abrir Test un artefacto JSON versionado y un tag Git anotado que fijen commit, hashes de código/configuraciones/manifiestos permitidos/checkpoints/contenedor/dependencias, TOP-3, márgenes, seeds, métricas, paleta y análisis; exigir gates pre-Test, repositorio limpio/publicado y pr |
| D56 | aprobaciones/ejecución | Test sellado: abrirlo una sola vez después de verificar D55 mediante un comando versionado en CEDIA; ejecutar sin pausas decisorias un DAG Slurm `afterok` para validación, YOLOv7/TOP-3 `ORIGINAL`, métricas/D54, síntesis MST de imagen completa/QC, **YOLOv7 independiente y TOP-3 por cada condición MST**, y D47–D49; separ |
| D57 | aprobaciones/ejecución | Roster de segmentadores: reutilizar sin borrar, reemplazar ni modificar sus checkpoints los mismos **15 segmentadores históricos más GrabCut**, volver a evaluar los 16 desde cero bajo D26–D32 y no heredar predicciones, métricas, ranking ni TOP-3 V1; una procedencia incompleta se documenta como limitación y no excluye a |
| D58 | aprobaciones/ejecución | Runbook CEDIA: usar `$HOME/Bench_Fairness_V2`, Git `main` para código, copias V2 no destructivas con `rsync -a --ignore-existing` para datos/checkpoints, SIF histórica verificada en solo lectura, `.cedia/venv`, jobs fijados a `compute-0-2`/exclusión de `compute-0-1`, recursos base 1×A100+32 CPU+60 GB≤48 h, arrays `%1`, |
| D59 | MST revisado | Reinicio MST aprobado después de los intentos 24269/24297: antes de la nueva ejecución borrar **solo las imágenes y resultados MST derivados de ambos intentos**, sin tocar originales, checkpoints, logs, ledger ni la bitácora; reconstruir desde cero sobre imagen completa, exigir `max(256, ceil(0.005 × área_imagen_comple |
| D60 | MST revisado | Reanudación eficiente MST aprobada durante 24336: conservar los PNG completos producidos por la **misma revisión científica**, recalcular automáticamente cada variante y reutilizarla solo si dimensiones y SHA-256 de píxeles decodificados coinciden; reemplazar atómicamente parciales inválidos, procesar cuatro fuentes co |

### Canon ejecutable completo

| Parámetro | Valor retenido |
|---|---|
| schema_version | 2 |
| seed | 20260828 |
| bootstrap_repetitions | 10000 |
| permutation_repetitions | 10000 |
| bbox_convention | xyxy_half_open |
| yolov7.repository | https://github.com/WongKinYiu/yolov7.git |
| yolov7.revision | a207844b1ce82d204ab36d87d496728d3d2348e7 |
| yolov7.checkpoint | yolov7_training.pt |
| yolov7.checkpoint_url | https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt |
| yolov7.image_size | 640 |
| yolov7.physical_batch | 32 |
| yolov7.accumulate | 2 |
| yolov7.epochs | 300 |
| yolov7.optimizer | SGD_Nesterov |
| yolov7.lr0 | 0.005 |
| yolov7.momentum | 0.9 |
| yolov7.weight_decay | 0.0005 |
| yolov7.warmup_epochs | 3 |
| yolov7.lrf | 0.1 |
| yolov7.confidence_threshold | 0.25 |
| yolov7.nms_iou_threshold | 0.45 |
| yolov7.multi_scale | False |
| yolov7.freeze_layers | 0 |
| hair.line_length_fraction | 0.035 |
| hair.line_thickness_fraction | 0.002 |
| hair.orientations_degrees | [0, 45, 90, 135] |
| hair.response_percentile | 90.0 |
| hair.minimum_elongation | 2.5 |
| hair.maximum_component_thickness_fraction | 0.025 |
| hair.dilation_fraction | 0.003 |
| hair.maximum_coverage_fraction | 0.18 |
| highlight.channel_minimum | 248 |
| highlight.channel_spread_maximum | 22 |
| minimum_clean_skin.pixels | 256 |
| minimum_clean_skin.area_fraction | 0.005 |
| ranking.primary | thresholded_jaccard_0.65 |
| ranking.top_k | 3 |
| ranking.tie_breakers | ['jaccard', 'dice', 'boundary_f1', 'negative_hd95_normalized'] |

| Condición MST | R | G | B |
|---|---|---|---|
| MST_01 | 246 | 237 | 228 |
| MST_02 | 243 | 231 | 219 |
| MST_03 | 247 | 234 | 208 |
| MST_04 | 234 | 218 | 186 |
| MST_05 | 215 | 189 | 150 |
| MST_06 | 160 | 126 | 86 |
| MST_07 | 130 | 92 | 67 |
| MST_08 | 96 | 65 | 52 |
| MST_09 | 58 | 49 | 42 |
| MST_10 | 41 | 36 | 32 |

### Definiciones matemáticas y símbolos que faltaban

Las siguientes ecuaciones hacen explícitas las operaciones ejecutadas. $\mathbb{1}[\cdot]$ es el indicador lógico (1 si la condición es verdadera); $|A|$ es cardinalidad; $\land$, $\lor$ y $\neg$ son AND, OR y NOT; $Q_p$ es el cuantil lineal $p$; $\lceil\cdot\rceil$ redondea hacia arriba; $\operatorname{median}$ es la mediana; $\operatorname{dist}$ es distancia euclidiana; $H,W$ son alto y ancho; $N$ es número de observaciones; $B$ es número de réplicas; $i$ identifica imagen, $m$ método, $k$ tono y $b$ réplica.

#### Entrenamiento, pérdidas y decisión YOLOv7

El batch efectivo fue $B_{ef}=B_{fis}\,A=32\times2=64$, donde $B_{fis}$ es el minibatch que reside en GPU y $A$ el número de acumulaciones antes de actualizar. Para SGD con Nesterov, una escritura equivalente de la actualización es $v_{t+1}=\mu v_t+\nabla L(\theta_t-\eta_t\mu v_t)$ y $\theta_{t+1}=\theta_t-\eta_t v_{t+1}$; $\theta$ son los pesos, $v$ la velocidad, $\mu=0.9$ el momentum y $\eta_t$ la tasa de aprendizaje. Se añadió penalización $L_2$ con `weight_decay` $5\times10^{-4}$. La tasa parte de $\eta_0=0.005$, usa tres epochs de calentamiento y termina en $\eta_f=0.1\eta_0=0.0005$ mediante One-Cycle cosenoidal. Esto implementa D18–D20 y conserva la transferencia completa discutida en [15]–[17].

La pérdida ejecutada fue `ComputeLossOTA`: $L=0.05L_{box}+0.30L_{cls}+0.70L_{obj}$, con asignación dinámica OTA, `fl_gamma=0` y `label_smoothing=0`; $L_{box}$ penaliza localización, $L_{cls}$ clase y $L_{obj}$ presencia de objeto. Una caja se conserva si $p(objeto)\ge0.25$ y NMS suprime la de menor confianza cuando $IoU(B_i,B_j)=|B_i\cap B_j|/|B_i\cup B_j|>0.45$. AutoAnchor se calcula solo con el fold de entrenamiento. Así, ninguna observación de validación/test decide anchors, gradientes o umbrales [2], [13].

#### Caja, ROI y fallback

La convención `xyxy_half_open` representa $B=[x_0,x_1)\times[y_0,y_1)$, por lo que $w=x_1-x_0$ y $h=y_1-y_0$. La expansión congelada es:

$$x'_0=\max(0,\lfloor x_0-mw\rfloor),\quad x'_1=\min(W,\lceil x_1+mw\rceil),$$
$$y'_0=\max(0,\lfloor y_0-mh\rfloor),\quad y'_1=\min(H,\lceil y_1+mh\rceil).$$

Aquí $m=0.1225400188$. Si el detector termina correctamente y no supera confianza 0.25/NMS 0.45, $ROI_i=[0,W_i)\times[0,H_i)$ y el estado es `valid_no_detection`; una excepción técnica es `failed` y obliga a corregir/repetir.

#### Métricas binarias y fronteras

$$Se=\frac{TP}{TP+FN},\quad Sp=\frac{TN}{TN+FP},\quad PPV=\frac{TP}{TP+FP},\quad Acc=\frac{TP+TN}{TP+TN+FP+FN}.$$

La frontera es $\partial M=M\land\neg\operatorname{erode}_{3\times3}(M)$. Con diagonal $d=\sqrt{H^2+W^2}$ y tolerancia $t=\max(1,0.01d)$, $P_b$ es la fracción de puntos de $\partial P$ a distancia $\le t$ de $\partial G$, $R_b$ la fracción recíproca y $BF1=2P_bR_b/(P_b+R_b)$. HD95 concatena las distancias dirigidas y toma $Q_{0.95}$; $HD95_n=HD95/d$. Si falta una frontera, BF1=0 y HD95=`NA`.

#### Pelo, reflejo, soporte y agregación cromática

Para $s=\min(H,W)$: longitud lineal $\ell=\operatorname{odd}(\max(5,\operatorname{round}(0.035s)))$, espesor $q=\max(1,\operatorname{round}(0.002s))$ y dilatación $d_h=\operatorname{odd}(\max(3,\operatorname{round}(0.003s)))$. En ángulos $\{0,45,90,135\}^{\circ}$ se aplica black-hat $BH_\theta=\operatorname{close}_{K_\theta}(I)-I$; el umbral es $\max(5,Q_{0.90}(BH))$. Se conserva un componente si longitud $\ge0.35\ell$, elongación $\ge2.5$ y espesor $\le0.025s$. Se marca alta cobertura cuando $|M_{hair}|/(HW)>0.18$ [6]–[12].

Los cuantiles 0.10 y 0.90 se calculan por canal. La media RGB recortada usa solo píxeles que satisfacen simultáneamente $Q_{0.10,c}\le p_c\le Q_{0.90,c}$ para $c\in\{R,G,B\}$. La mediana Lab y el ITA se obtienen del conjunto no recortado. Esta separación evita confundir una media robusta descriptiva con el estimador ITA [24], [25], [29].

#### Inversa Lab, gamut y CIEDE2000

Con $f_y=(L^*+16)/116$, $f_x=f_y+a^*/500$, $f_z=f_y-b^*/200$ e $\delta=6/29$:

$$f^{-1}(u)=\begin{cases}u^3,&u>\delta,\\3\delta^2(u-4/29),&u\le\delta.\end{cases}$$

Luego $(X,Y,Z)=(0.95047f^{-1}(f_x),f^{-1}(f_y),1.08883f^{-1}(f_z))$ y se aplica la matriz XYZ→RGB lineal inversa de sRGB. Para cada canal $c_l$: $c_s=12.92c_l$ si $c_l\le0.0031308$; de otro modo $c_s=1.055\max(c_l,0)^{1/2.4}-0.055$. Finalmente $c_8=\operatorname{round}(255\operatorname{clip}(c_s,0,1))$. La fracción de clipping cuenta canales con $c_s\le0$ o $c_s\ge1$.

Para CIEDE2000 [32], sean dos colores $(L'_1,a'_1,b'_1)$ y $(L'_2,a'_2,b'_2)$. Se calculan $C_i=\sqrt{a_i^2+b_i^2}$, $\bar C=(C_1+C_2)/2$, $G=\tfrac12[1-\sqrt{\bar C^7/(\bar C^7+25^7)}]$, $a'_i=(1+G)a_i$, $C'_i=\sqrt{{a'_i}^2+b_i^2}$ y $h'_i=\operatorname{atan2}(b_i,a'_i)$ llevado a $[0,360)$ grados. A partir de ellos:

$$\Delta L'=L'_2-L'_1,\quad \Delta C'=C'_2-C'_1,\quad \Delta H'=2\sqrt{C'_1C'_2}\sin(\Delta h'/2).$$

El ángulo $\Delta h'$ es la diferencia mínima circular; vale cero si $C'_1C'_2=0$. Con promedios $\bar L'$, $\bar C'$ y $\bar h'$ —este último también circular—:

$$T=1-0.17\cos(\bar h'-30)+0.24\cos(2\bar h')+0.32\cos(3\bar h'+6)-0.20\cos(4\bar h'-63),$$

$$S_L=1+\frac{0.015(\bar L'-50)^2}{\sqrt{20+(\bar L'-50)^2}},\quad S_C=1+0.045\bar C',\quad S_H=1+0.015\bar C'T,$$

$$R_T=-2\sqrt{\frac{\bar C'^7}{\bar C'^7+25^7}}\sin\left[60\exp\left(-\left(\frac{\bar h'-275}{25}\right)^2\right)\right],$$

$$\Delta E_{00}=\sqrt{\left(\frac{\Delta L'}{S_L}\right)^2+\left(\frac{\Delta C'}{S_C}\right)^2+\left(\frac{\Delta H'}{S_H}\right)^2+R_T\left(\frac{\Delta C'}{S_C}\right)\left(\frac{\Delta H'}{S_H}\right)}.$$

Los factores paramétricos son $k_L=k_C=k_H=1$, omitidos en la escritura anterior. Los argumentos trigonométricos se evalúan en grados según el algoritmo de referencia. $\Delta E_{00}$ fue QC continuo de fidelidad cromática, no criterio de exclusión ni variable primaria MST.

#### Bootstrap, permutación, Holm, McNemar y orden

En bootstrap pareado, un vector común $I_b=(I_{b1},\ldots,I_{bN})$ remuestrea imágenes con reemplazo y $\bar\Delta_b=N^{-1}\sum_j(x_{I_{bj}}-y_{I_{bj}})$. El IC95% es $[Q_{0.025}(\bar\Delta),Q_{0.975}(\bar\Delta)]$. En permutación pareada, signos $s_{bi}\in\{-1,+1\}$ producen $T_b=|N^{-1}\sum_i s_{bi}\Delta_i|$ y $p=(1+\sum_b\mathbb{1}[T_b\ge|\bar\Delta|])/(B+1)$. Holm ordena los $p$ y usa $p^{adj}_{(r)}=\max_{j\le r}\min(1,(M-j+1)p_{(j)})$. McNemar exacto usa los discordantes $b,c$ y una binomial $Bin(b+c,0.5)$. Estas inferencias se aplicaron a Validation; MST quedó descriptivo por diseño [21], [22].

#### Correlación y acuerdo

Spearman es la correlación de Pearson entre rangos; Kendall $\tau_b=(C-D)/\sqrt{(C+D+T_x)(C+D+T_y)}$, donde $C,D$ son pares concordantes/discordantes y $T_x,T_y$ empates exclusivos. El ICC usado es de acuerdo absoluto y sus réplicas remuestrean pacientes completos. Correlación no sustituye acuerdo: Bias y límites de Bland–Altman describen diferencias, mientras ICC describe acuerdo absoluto [35], [36].

### Matriz de ramas operativas

| Punto | Condición | Acción reproducible | Resultado/denominador |
|---|---|---|---|
| YOLOv7 ORIGINAL o MST | ejecución válida con caja | escoger la caja de mayor confianza, expandir con $m=0.1225400188$ y recortar a imagen | `detected`; ROI dinámica |
| YOLOv7 ORIGINAL o MST | ejecución válida sin caja | usar toda la imagen de esa condición | `valid_no_detection`; no es error |
| YOLOv7 | excepción, archivo ilegible o salida inválida | detener el job, diagnosticar y reejecutar | `failed`; nunca se imputa |
| Segmentador | salida válida no vacía | restaurar a ROI según adaptador, sin morfología común | entra a métricas y color |
| Segmentador | salida válida vacía/llena | conservar máscara científica y marcar degeneración | entra a métricas; color=`NA` |
| Segmentador | excepción técnica | detener/corregir/repetir la revisión completa | no entra al ranking incompleto |
| Pelo | cobertura $\le0.18$ o $>0.18$ | conservar siempre máscara; en el segundo caso añadir flag | píxeles excluidos del soporte |
| Piel limpia | lesión válida y $\lvert C\rvert\ge\max(256,\lceil0.005HW\rceil)$ | calcular RGB recortado, mediana Lab e ITA | `available` |
| Piel limpia | lesión degenerada | registrar causa sin fallback espacial | `unavailable_degenerate_lesion_mask` |
| Piel limpia | soporte menor al mínimo | registrar causa sin resumir color | `unavailable_insufficient_clean_skin` |
| Piel limpia | recorte multicanal vacío | conservar Lab/ITA si existen; RGB recortado no disponible | `unavailable_empty_trimmed_set` para esa salida |
| MST | soporte fuente suficiente | trasladar $L^*,a^*,b^*$ y sintetizar imagen completa | condición evaluable |
| MST | soporte fuente insuficiente | no sintetizar ni inventar color; registrar no disponibilidad | denominador común final: 975/1000 |
| MSKCC continuo | referencia o predicción ausente | mantener `NA`; no imputar ni fusionar estratos | $N$ específico por método/estrato |
| MSKCC ordinal | etiqueta MST disponible | Kendall $\tau_b$ y Spearman $\rho$ | análisis de orden, no acuerdo absoluto |

La matriz es importante porque separa una ausencia científicamente válida de un fallo de software. Solo `valid_no_detection` activa el fallback de imagen completa; ninguna excepción técnica se convierte silenciosamente en no-detección. Del mismo modo, `NA` preserva el denominador real y evita crear evidencia cromática inexistente [19], [31], [34].

## Resultados exhaustivos y resúmenes por rama

### Roster completo y decisión nativa de los 16 métodos

Los 15 checkpoints se cargaron con verificación SHA-256 y `strict=True`; GrabCut no tiene checkpoint. “Decisión” indica cómo se obtuvo la máscara binaria antes de restaurarla a la geometría ROI. Las interpolaciones de entrada son nativas del adaptador; la restauración de máscaras binarias usa vecino más cercano, salvo las familias Unixio, que restauran probabilidad bilineal antes de umbralizar. Esta heterogeneidad se preservó deliberadamente por D26: homogeneizarla habría cambiado los modelos que se comparaban.

| Método | Arquitectura/checkpoint | Entrada | Normalización/decisión nativa | Resultado V2 |
|---|---|---:|---|---|
| AViT | ViT-B AViT, ISIC | 224² | Normalización del repositorio; `sigmoid ≥ 0.5` | TOP-1 |
| UltraLight VM-UNet | VMamba ultraligera | 256² | Normalización nativa; binarización nativa | Rank 8 |
| BA-Transformer | Transformer guiado por borde, ISIC 2016 | 352² | Normalización nativa; `sigmoid ≥ 0.5` | Rank 13 |
| U-Net/ResNet34 | U-Net con encoder ResNet34, ISIC 2018 | 256² | ImageNet; `sigmoid > 0.5` | Rank 7 |
| SkinMamba ISIC17 | CNN/Mamba con guía de borde | 224² | Estadísticos ISIC17 + reescala nativa | Rank 15; máscara vacía |
| SkinMamba ISIC18 | CNN/Mamba con guía de borde | 224² | Estadísticos ISIC18 + reescala nativa | Rank 16; máscara vacía |
| Unixio U-Net | U-Net/ResNet34 | 256² | `/255`; probabilidad bilineal; `>0.5` | Rank 6 |
| Unixio U-Net++ | U-Net++/ResNet34 | 256² | `/255`; probabilidad bilineal; `>0.5` | Rank 5 |
| Theodore SegFormer | SegFormer-B0, dos clases | 128² | ImageNet; `argmax` de logits | Rank 10 |
| VM-UNet ISIC18 | VMamba/VM-UNet | 256² | Estadísticos ISIC18 + reescala nativa | Rank 4 |
| Unixio Attention U-Net | Attention U-Net | 256² | `/255`; probabilidad bilineal; `>0.5` | Rank 9 |
| Theodore U-Net | U-Net compacta, dos clases | 128² | `/255`; `argmax` de logits | Rank 11 |
| Theodore Inception | Segmentador Inception, dos clases | 128² | `/255`; `argmax` de logits | Rank 14 |
| VM-UNet ISIC17 | VMamba/VM-UNet | 256² | Estadísticos ISIC17 + reescala nativa | TOP-3 |
| DeLightSAM-Dermoscopy | ESP-MedSAM/TinyViT, dermoscopia | 1024² | Normalización nativa; `sigmoid ≥ 0.5` | TOP-2 |
| GrabCut | OpenCV GMM + corte de grafo | ROI nativa | Rectángulo interior de 1 px; 5 iteraciones; FG=`FG ∪ probable-FG` | Rank 12 |

GrabCut minimiza una energía de etiquetas $\mathbf z$ compuesta por ajuste a modelos de mezcla gaussiana de fondo/primer plano y suavidad entre píxeles vecinos:

$$E(\mathbf z,\boldsymbol\theta,I)=U(\mathbf z,\boldsymbol\theta,I)+V(\mathbf z,I).$$

$U$ penaliza asignaciones cromáticamente improbables bajo los GMM parametrizados por $\boldsymbol\theta$; $V$ penaliza discontinuidades entre vecinos con colores parecidos. El corte mínimo alterna estimación de GMM y etiquetas durante cinco iteraciones. La fórmula explica el algoritmo, pero el resultado reproducible está definido por OpenCV y su versión registrada, no por una reimplementación propia.

### Validation: todas las métricas de los 16 métodos

| Rank | Método | N | TJ | J | Dice | Se | Sp | PPV | Acc | BF1 | HD95 px | HD95 norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | avit | 100 | 0.778707 | 0.818906 | 0.886077 | 0.937530 | 0.945166 | 0.857157 | 0.945655 | 0.520440 | 183.368055 | 0.055807 |
| 2 | delightsam-dermoscopy | 100 | 0.767342 | 0.807620 | 0.878807 | 0.930882 | 0.944751 | 0.854090 | 0.945139 | 0.443103 | 182.971362 | 0.056906 |
| 3 | vmunet-isic17 | 100 | 0.766592 | 0.804662 | 0.874156 | 0.907689 | 0.958099 | 0.874189 | 0.940680 | 0.546207 | 202.999173 | 0.059820 |
| 4 | vmunet-isic18 | 100 | 0.766410 | 0.807226 | 0.877667 | 0.940177 | 0.944687 | 0.843921 | 0.943965 | 0.504294 | 201.780744 | 0.059973 |
| 5 | unixio-unetpp-isic2018 | 100 | 0.748326 | 0.796480 | 0.873575 | 0.929262 | 0.941604 | 0.843591 | 0.936342 | 0.445732 | 216.449343 | 0.066835 |
| 6 | unixio-unet-isic2018 | 100 | 0.742891 | 0.800877 | 0.875055 | 0.930378 | 0.943457 | 0.846525 | 0.936310 | 0.478866 | 190.769444 | 0.058336 |
| 7 | unet-resnet34-isic2018 | 100 | 0.736601 | 0.780215 | 0.862290 | 0.945460 | 0.922854 | 0.810229 | 0.932462 | 0.396531 | 231.973677 | 0.068192 |
| 8 | ultralight-vm-unet | 100 | 0.730754 | 0.788049 | 0.859017 | 0.878810 | 0.955625 | 0.881699 | 0.930075 | 0.563149 | 209.316719 | 0.061969 |
| 9 | unixio-attention-unet-isic2018 | 100 | 0.713832 | 0.779530 | 0.858454 | 0.892828 | 0.953711 | 0.859258 | 0.927107 | 0.461087 | 203.861107 | 0.061232 |
| 10 | theodore-segformer-isic2018 | 100 | 0.675566 | 0.746751 | 0.835184 | 0.896182 | 0.941001 | 0.824655 | 0.919079 | 0.334474 | 237.937662 | 0.070535 |
| 11 | theodore-unet-isic2018 | 100 | 0.650087 | 0.722651 | 0.810663 | 0.854941 | 0.943677 | 0.808632 | 0.905021 | 0.315990 | 248.003596 | 0.070949 |
| 12 | grabcut | 100 | 0.642566 | 0.751893 | 0.837155 | 0.899209 | 0.932711 | 0.820917 | 0.910246 | 0.426466 | 279.108786 | 0.080542 |
| 13 | ba-transformer | 100 | 0.581283 | 0.682900 | 0.768395 | 0.732195 | 0.977107 | 0.893143 | 0.895812 | 0.416932 | 270.794226 | 0.083289 |
| 14 | theodore-inception-isic2018 | 100 | 0.430923 | 0.610483 | 0.716978 | 0.694706 | 0.936631 | 0.850112 | 0.866587 | 0.352027 | 349.449045 | 0.108712 |
| 15 | skinmamba-isic17 | 100 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.731519 | 0.000000 | NA | NA |
| 16 | skinmamba-isic18 | 100 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.731519 | 0.000000 | NA | NA |

### Validation: las 120 comparaciones pareadas de TJ

Cada fila es `primero − segundo`; `p_raw` proviene de permutación pareada y `p_Holm` corrige las 120 comparaciones. Incluir la matriz completa evita seleccionar únicamente contrastes favorables.

| Primero | Segundo | ΔTJ | IC95% | p_raw | p_Holm |
|---|---|---|---|---|---|
| avit | delightsam-dermoscopy | 0.011365 | [-0.010475, 0.035738] | 0.419658 | 1.000000 |
| avit | vmunet-isic17 | 0.012114 | [-0.026275, 0.052603] | 0.533047 | 1.000000 |
| avit | vmunet-isic18 | 0.012296 | [-0.029017, 0.056166] | 0.545445 | 1.000000 |
| avit | unixio-unetpp-isic2018 | 0.030380 | [-0.003563, 0.066486] | 0.086591 | 1.000000 |
| avit | unixio-unet-isic2018 | 0.035816 | [0.001595, 0.076009] | 0.057994 | 1.000000 |
| avit | unet-resnet34-isic2018 | 0.042106 | [0.004289, 0.082453] | 0.032497 | 1.000000 |
| avit | ultralight-vm-unet | 0.047953 | [-0.006852, 0.106822] | 0.104790 | 1.000000 |
| avit | unixio-attention-unet-isic2018 | 0.064874 | [0.014921, 0.120604] | 0.014299 | 0.614839 |
| avit | theodore-segformer-isic2018 | 0.103140 | [0.051900, 0.159349] | 0.000400 | 0.022398 |
| avit | theodore-unet-isic2018 | 0.128620 | [0.069610, 0.191523] | 0.000200 | 0.012999 |
| avit | grabcut | 0.136140 | [0.085552, 0.192989] | 0.000100 | 0.011999 |
| avit | ba-transformer | 0.197424 | [0.124795, 0.272155] | 0.000100 | 0.011999 |
| avit | theodore-inception-isic2018 | 0.347783 | [0.267532, 0.426318] | 0.000100 | 0.011999 |
| avit | skinmamba-isic17 | 0.778707 | [0.723406, 0.829504] | 0.000100 | 0.011999 |
| avit | skinmamba-isic18 | 0.778707 | [0.723406, 0.829504] | 0.000100 | 0.011999 |
| delightsam-dermoscopy | vmunet-isic17 | 0.000749 | [-0.030266, 0.034151] | 0.976802 | 1.000000 |
| delightsam-dermoscopy | vmunet-isic18 | 0.000931 | [-0.035184, 0.038262] | 0.975802 | 1.000000 |
| delightsam-dermoscopy | unixio-unetpp-isic2018 | 0.019015 | [-0.006994, 0.047719] | 0.204380 | 1.000000 |
| delightsam-dermoscopy | unixio-unet-isic2018 | 0.024451 | [-0.001613, 0.056655] | 0.126787 | 1.000000 |
| delightsam-dermoscopy | unet-resnet34-isic2018 | 0.030741 | [-0.002400, 0.066644] | 0.075892 | 1.000000 |
| delightsam-dermoscopy | ultralight-vm-unet | 0.036588 | [-0.013828, 0.090818] | 0.184282 | 1.000000 |
| delightsam-dermoscopy | unixio-attention-unet-isic2018 | 0.053509 | [0.009477, 0.106207] | 0.027097 | 1.000000 |
| delightsam-dermoscopy | theodore-segformer-isic2018 | 0.091775 | [0.045592, 0.143111] | 0.000300 | 0.017698 |
| delightsam-dermoscopy | theodore-unet-isic2018 | 0.117254 | [0.063450, 0.176558] | 0.000100 | 0.011999 |
| delightsam-dermoscopy | grabcut | 0.124775 | [0.071671, 0.182582] | 0.000100 | 0.011999 |
| delightsam-dermoscopy | ba-transformer | 0.186059 | [0.111663, 0.262132] | 0.000100 | 0.011999 |
| delightsam-dermoscopy | theodore-inception-isic2018 | 0.336418 | [0.253928, 0.417311] | 0.000100 | 0.011999 |
| delightsam-dermoscopy | skinmamba-isic17 | 0.767342 | [0.711481, 0.815786] | 0.000100 | 0.011999 |
| delightsam-dermoscopy | skinmamba-isic18 | 0.767342 | [0.711481, 0.815786] | 0.000100 | 0.011999 |
| vmunet-isic17 | vmunet-isic18 | 0.000182 | [-0.037489, 0.039035] | 0.988701 | 1.000000 |
| vmunet-isic17 | unixio-unetpp-isic2018 | 0.018266 | [-0.014623, 0.050140] | 0.301070 | 1.000000 |
| vmunet-isic17 | unixio-unet-isic2018 | 0.023702 | [-0.008717, 0.058040] | 0.159684 | 1.000000 |
| vmunet-isic17 | unet-resnet34-isic2018 | 0.029992 | [-0.008837, 0.068659] | 0.140286 | 1.000000 |
| vmunet-isic17 | ultralight-vm-unet | 0.035838 | [-0.002343, 0.076877] | 0.088191 | 1.000000 |
| vmunet-isic17 | unixio-attention-unet-isic2018 | 0.052760 | [0.014679, 0.095766] | 0.010099 | 0.454455 |
| vmunet-isic17 | theodore-segformer-isic2018 | 0.091026 | [0.057188, 0.130730] | 0.000100 | 0.011999 |
| vmunet-isic17 | theodore-unet-isic2018 | 0.116505 | [0.067885, 0.166871] | 0.000100 | 0.011999 |
| vmunet-isic17 | grabcut | 0.124026 | [0.068083, 0.183839] | 0.000100 | 0.011999 |
| vmunet-isic17 | ba-transformer | 0.185310 | [0.114892, 0.257011] | 0.000100 | 0.011999 |
| vmunet-isic17 | theodore-inception-isic2018 | 0.335669 | [0.256041, 0.414136] | 0.000100 | 0.011999 |
| vmunet-isic17 | skinmamba-isic17 | 0.766592 | [0.710570, 0.817170] | 0.000100 | 0.011999 |
| vmunet-isic17 | skinmamba-isic18 | 0.766592 | [0.710570, 0.817170] | 0.000100 | 0.011999 |
| vmunet-isic18 | unixio-unetpp-isic2018 | 0.018084 | [-0.023177, 0.059224] | 0.399860 | 1.000000 |
| vmunet-isic18 | unixio-unet-isic2018 | 0.023520 | [-0.013319, 0.063882] | 0.230477 | 1.000000 |
| vmunet-isic18 | unet-resnet34-isic2018 | 0.029810 | [-0.006776, 0.067179] | 0.114589 | 1.000000 |
| vmunet-isic18 | ultralight-vm-unet | 0.035656 | [-0.017650, 0.089881] | 0.212779 | 1.000000 |
| vmunet-isic18 | unixio-attention-unet-isic2018 | 0.052578 | [0.004567, 0.104241] | 0.035096 | 1.000000 |
| vmunet-isic18 | theodore-segformer-isic2018 | 0.090844 | [0.047238, 0.138342] | 0.000300 | 0.017698 |
| vmunet-isic18 | theodore-unet-isic2018 | 0.116323 | [0.055480, 0.179421] | 0.000600 | 0.032397 |
| vmunet-isic18 | grabcut | 0.123844 | [0.069040, 0.183507] | 0.000100 | 0.011999 |
| vmunet-isic18 | ba-transformer | 0.185128 | [0.107530, 0.262509] | 0.000100 | 0.011999 |
| vmunet-isic18 | theodore-inception-isic2018 | 0.335487 | [0.255245, 0.416382] | 0.000100 | 0.011999 |
| vmunet-isic18 | skinmamba-isic17 | 0.766410 | [0.709496, 0.817401] | 0.000100 | 0.011999 |
| vmunet-isic18 | skinmamba-isic18 | 0.766410 | [0.709496, 0.817401] | 0.000100 | 0.011999 |
| unixio-unetpp-isic2018 | unixio-unet-isic2018 | 0.005436 | [-0.024291, 0.038526] | 0.725227 | 1.000000 |
| unixio-unetpp-isic2018 | unet-resnet34-isic2018 | 0.011725 | [-0.025064, 0.047592] | 0.571143 | 1.000000 |
| unixio-unetpp-isic2018 | ultralight-vm-unet | 0.017572 | [-0.025725, 0.063274] | 0.460454 | 1.000000 |
| unixio-unetpp-isic2018 | unixio-attention-unet-isic2018 | 0.034494 | [-0.008920, 0.082645] | 0.151885 | 1.000000 |
| unixio-unetpp-isic2018 | theodore-segformer-isic2018 | 0.072760 | [0.030920, 0.118730] | 0.001000 | 0.051995 |
| unixio-unetpp-isic2018 | theodore-unet-isic2018 | 0.098239 | [0.047681, 0.152247] | 0.000200 | 0.012999 |
| unixio-unetpp-isic2018 | grabcut | 0.105760 | [0.057033, 0.160109] | 0.000100 | 0.011999 |
| unixio-unetpp-isic2018 | ba-transformer | 0.167043 | [0.100838, 0.236865] | 0.000100 | 0.011999 |
| unixio-unetpp-isic2018 | theodore-inception-isic2018 | 0.317403 | [0.241366, 0.393837] | 0.000100 | 0.011999 |
| unixio-unetpp-isic2018 | skinmamba-isic17 | 0.748326 | [0.691884, 0.799307] | 0.000100 | 0.011999 |
| unixio-unetpp-isic2018 | skinmamba-isic18 | 0.748326 | [0.691884, 0.799307] | 0.000100 | 0.011999 |
| unixio-unet-isic2018 | unet-resnet34-isic2018 | 0.006290 | [-0.038348, 0.048819] | 0.770923 | 1.000000 |
| unixio-unet-isic2018 | ultralight-vm-unet | 0.012137 | [-0.035492, 0.062181] | 0.630237 | 1.000000 |
| unixio-unet-isic2018 | unixio-attention-unet-isic2018 | 0.029058 | [-0.011509, 0.073412] | 0.177782 | 1.000000 |
| unixio-unet-isic2018 | theodore-segformer-isic2018 | 0.067324 | [0.021630, 0.115798] | 0.004600 | 0.220778 |
| unixio-unet-isic2018 | theodore-unet-isic2018 | 0.092804 | [0.045386, 0.145016] | 0.000200 | 0.012999 |
| unixio-unet-isic2018 | grabcut | 0.100324 | [0.049832, 0.155134] | 0.000200 | 0.012999 |
| unixio-unet-isic2018 | ba-transformer | 0.161608 | [0.091330, 0.232798] | 0.000200 | 0.012999 |
| unixio-unet-isic2018 | theodore-inception-isic2018 | 0.311967 | [0.233932, 0.389696] | 0.000100 | 0.011999 |
| unixio-unet-isic2018 | skinmamba-isic17 | 0.742891 | [0.681626, 0.797091] | 0.000100 | 0.011999 |
| unixio-unet-isic2018 | skinmamba-isic18 | 0.742891 | [0.681626, 0.797091] | 0.000100 | 0.011999 |
| unet-resnet34-isic2018 | ultralight-vm-unet | 0.005847 | [-0.047350, 0.060173] | 0.835816 | 1.000000 |
| unet-resnet34-isic2018 | unixio-attention-unet-isic2018 | 0.022768 | [-0.029345, 0.077742] | 0.418858 | 1.000000 |
| unet-resnet34-isic2018 | theodore-segformer-isic2018 | 0.061034 | [0.016338, 0.108911] | 0.008599 | 0.395560 |
| unet-resnet34-isic2018 | theodore-unet-isic2018 | 0.086514 | [0.028085, 0.145142] | 0.002500 | 0.122488 |
| unet-resnet34-isic2018 | grabcut | 0.094034 | [0.039597, 0.153166] | 0.001300 | 0.066293 |
| unet-resnet34-isic2018 | ba-transformer | 0.155318 | [0.077210, 0.234575] | 0.000500 | 0.027497 |
| unet-resnet34-isic2018 | theodore-inception-isic2018 | 0.305677 | [0.226574, 0.384483] | 0.000100 | 0.011999 |
| unet-resnet34-isic2018 | skinmamba-isic17 | 0.736601 | [0.680805, 0.786457] | 0.000100 | 0.011999 |
| unet-resnet34-isic2018 | skinmamba-isic18 | 0.736601 | [0.680805, 0.786457] | 0.000100 | 0.011999 |
| ultralight-vm-unet | unixio-attention-unet-isic2018 | 0.016922 | [-0.027236, 0.061586] | 0.472553 | 1.000000 |
| ultralight-vm-unet | theodore-segformer-isic2018 | 0.055187 | [0.015500, 0.095252] | 0.007999 | 0.375962 |
| ultralight-vm-unet | theodore-unet-isic2018 | 0.080667 | [0.034835, 0.128315] | 0.000700 | 0.037096 |
| ultralight-vm-unet | grabcut | 0.088188 | [0.031811, 0.146012] | 0.002300 | 0.114989 |
| ultralight-vm-unet | ba-transformer | 0.149471 | [0.086866, 0.214699] | 0.000100 | 0.011999 |
| ultralight-vm-unet | theodore-inception-isic2018 | 0.299831 | [0.228911, 0.372467] | 0.000100 | 0.011999 |
| ultralight-vm-unet | skinmamba-isic17 | 0.730754 | [0.664271, 0.791620] | 0.000100 | 0.011999 |
| ultralight-vm-unet | skinmamba-isic18 | 0.730754 | [0.664271, 0.791620] | 0.000100 | 0.011999 |
| unixio-attention-unet-isic2018 | theodore-segformer-isic2018 | 0.038266 | [-0.004531, 0.079961] | 0.078692 | 1.000000 |
| unixio-attention-unet-isic2018 | theodore-unet-isic2018 | 0.063745 | [0.029161, 0.100972] | 0.000100 | 0.011999 |
| unixio-attention-unet-isic2018 | grabcut | 0.071266 | [0.016135, 0.128954] | 0.016798 | 0.705529 |
| unixio-attention-unet-isic2018 | ba-transformer | 0.132550 | [0.070035, 0.196152] | 0.000300 | 0.017698 |
| unixio-attention-unet-isic2018 | theodore-inception-isic2018 | 0.282909 | [0.206554, 0.359041] | 0.000100 | 0.011999 |
| unixio-attention-unet-isic2018 | skinmamba-isic17 | 0.713832 | [0.647720, 0.772385] | 0.000100 | 0.011999 |
| unixio-attention-unet-isic2018 | skinmamba-isic18 | 0.713832 | [0.647720, 0.772385] | 0.000100 | 0.011999 |
| theodore-segformer-isic2018 | theodore-unet-isic2018 | 0.025479 | [-0.013332, 0.067845] | 0.215378 | 1.000000 |
| theodore-segformer-isic2018 | grabcut | 0.033000 | [-0.027126, 0.099528] | 0.306769 | 1.000000 |
| theodore-segformer-isic2018 | ba-transformer | 0.094284 | [0.024903, 0.164722] | 0.010899 | 0.479552 |
| theodore-segformer-isic2018 | theodore-inception-isic2018 | 0.244643 | [0.165027, 0.325965] | 0.000100 | 0.011999 |
| theodore-segformer-isic2018 | skinmamba-isic17 | 0.675566 | [0.609259, 0.735905] | 0.000100 | 0.011999 |
| theodore-segformer-isic2018 | skinmamba-isic18 | 0.675566 | [0.609259, 0.735905] | 0.000100 | 0.011999 |
| theodore-unet-isic2018 | grabcut | 0.007521 | [-0.052715, 0.069911] | 0.817118 | 1.000000 |
| theodore-unet-isic2018 | ba-transformer | 0.068804 | [0.007782, 0.132481] | 0.035196 | 1.000000 |
| theodore-unet-isic2018 | theodore-inception-isic2018 | 0.219164 | [0.142554, 0.295857] | 0.000100 | 0.011999 |
| theodore-unet-isic2018 | skinmamba-isic17 | 0.650087 | [0.580454, 0.713250] | 0.000100 | 0.011999 |
| theodore-unet-isic2018 | skinmamba-isic18 | 0.650087 | [0.580454, 0.713250] | 0.000100 | 0.011999 |
| grabcut | ba-transformer | 0.061284 | [-0.007116, 0.130134] | 0.089291 | 1.000000 |
| grabcut | theodore-inception-isic2018 | 0.211643 | [0.141021, 0.283475] | 0.000100 | 0.011999 |
| grabcut | skinmamba-isic17 | 0.642566 | [0.568353, 0.712228] | 0.000100 | 0.011999 |
| grabcut | skinmamba-isic18 | 0.642566 | [0.568353, 0.712228] | 0.000100 | 0.011999 |
| ba-transformer | theodore-inception-isic2018 | 0.150359 | [0.076078, 0.225768] | 0.000200 | 0.012999 |
| ba-transformer | skinmamba-isic17 | 0.581283 | [0.500826, 0.657689] | 0.000100 | 0.011999 |
| ba-transformer | skinmamba-isic18 | 0.581283 | [0.500826, 0.657689] | 0.000100 | 0.011999 |
| theodore-inception-isic2018 | skinmamba-isic17 | 0.430923 | [0.349935, 0.513267] | 0.000100 | 0.011999 |
| theodore-inception-isic2018 | skinmamba-isic18 | 0.430923 | [0.349935, 0.513267] | 0.000100 | 0.011999 |
| skinmamba-isic17 | skinmamba-isic18 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |

### Validation: disponibilidad colorimétrica por método TOP-3

| Método | available | degenerate | insufficient | trimmed-empty | Total |
|---|---|---|---|---|---|
| avit | 100 | 0 | 0 | 0 | 100 |
| delightsam-dermoscopy | 100 | 0 | 0 | 0 | 100 |
| vmunet-isic17 | 99 | 0 | 1 | 0 | 100 |

### Test original: todas las métricas del TOP-3

| Método | N | TJ | J | Dice | Se | Sp | PPV | Acc | BF1 | HD95 px | HD95 norm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| avit | 1000 | 0.766669 | 0.809185 | 0.880602 | 0.938563 | 0.932968 | 0.858766 | 0.935530 | 0.498894 | 226.145034 | 0.062209 |
| delightsam-dermoscopy | 1000 | 0.742512 | 0.791406 | 0.869111 | 0.923753 | 0.936752 | 0.853147 | 0.929580 | 0.435713 | 221.088651 | 0.062970 |
| vmunet-isic17 | 1000 | 0.754465 | 0.798340 | 0.869385 | 0.907274 | 0.947366 | 0.874426 | 0.929293 | 0.529279 | 216.385105 | 0.060656 |

### MST: tabla completa de las 30 combinaciones método–tono

Para cada métrica se reportan media, mediana, Q1, Q3, IC95% pareado y peor delta individual. `Transiciones` cuenta ORIGINAL→MST (`detected→detected`, `detected→no`, `no→detected`, `no→no`).

| Método | MST | N | Det.rate | Transiciones | ΔJ mean | ΔJ med | ΔJ Q1 | ΔJ Q3 | ΔJ IC95 | ΔJ peor | ΔTJ mean | ΔTJ med | ΔTJ Q1 | ΔTJ Q3 | ΔTJ IC95 | ΔTJ peor | ΔDice mean | ΔDice med | ΔDice Q1 | ΔDice Q3 | ΔDice IC95 | ΔDice peor | ΔBF1 mean | ΔBF1 med | ΔBF1 Q1 | ΔBF1 Q3 | ΔBF1 IC95 | ΔBF1 peor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| avit | MST_01 | 975 | 0.964103 | {'detected->detected': 929, 'detected->valid_no_detection': 30, 'valid_no_detection->detected': 11, 'valid_no_detection->valid_no_detection': 5} | -0.022882 | -0.007102 | -0.025973 | 0.007449 | [-0.031531, -0.014775] | -0.952852 | -0.031826 | -0.004584 | -0.024435 | 0.004482 | [-0.044328, -0.019920] | -0.966335 | -0.019648 | -0.004165 | -0.015668 | 0.004260 | [-0.028598, -0.011265] | -0.975857 | -0.052024 | -0.015039 | -0.105633 | 0.013918 | [-0.061620, -0.042510] | -0.830940 |
| avit | MST_02 | 975 | 0.972308 | {'detected->detected': 937, 'detected->valid_no_detection': 22, 'valid_no_detection->detected': 11, 'valid_no_detection->valid_no_detection': 5} | -0.017845 | -0.003963 | -0.021640 | 0.009662 | [-0.025727, -0.010333] | -0.952852 | -0.023931 | -0.001955 | -0.020273 | 0.007896 | [-0.036332, -0.011785] | -0.966335 | -0.015594 | -0.002260 | -0.013096 | 0.005626 | [-0.023442, -0.007986] | -0.975857 | -0.037939 | -0.009923 | -0.083247 | 0.018327 | [-0.047130, -0.029072] | -0.841547 |
| avit | MST_03 | 975 | 0.978462 | {'detected->detected': 943, 'detected->valid_no_detection': 16, 'valid_no_detection->detected': 11, 'valid_no_detection->valid_no_detection': 5} | -0.014838 | -0.005672 | -0.022896 | 0.008905 | [-0.022340, -0.007350] | -0.952852 | -0.019985 | -0.003881 | -0.021042 | 0.006494 | [-0.031794, -0.008561] | -0.966335 | -0.011546 | -0.003386 | -0.014104 | 0.005193 | [-0.019098, -0.003943] | -0.975857 | -0.042188 | -0.013551 | -0.098147 | 0.017849 | [-0.051209, -0.033576] | -0.791203 |
| avit | MST_04 | 975 | 0.981538 | {'detected->detected': 950, 'detected->valid_no_detection': 9, 'valid_no_detection->detected': 7, 'valid_no_detection->valid_no_detection': 9} | -0.012481 | -0.004326 | -0.017252 | 0.005859 | [-0.019438, -0.005825] | -0.952852 | -0.019098 | -0.002589 | -0.015869 | 0.004284 | [-0.029617, -0.008949] | -0.966335 | -0.010079 | -0.002369 | -0.010479 | 0.003605 | [-0.017242, -0.003254] | -0.975857 | -0.035986 | -0.011774 | -0.075553 | 0.010530 | [-0.043586, -0.028743] | -0.791203 |
| avit | MST_05 | 975 | 0.979487 | {'detected->detected': 949, 'detected->valid_no_detection': 10, 'valid_no_detection->detected': 6, 'valid_no_detection->valid_no_detection': 10} | -0.013679 | -0.004643 | -0.015040 | 0.003038 | [-0.019955, -0.007690] | -0.952852 | -0.021365 | -0.003400 | -0.013981 | 0.002176 | [-0.031070, -0.012217] | -0.966335 | -0.011028 | -0.002826 | -0.009074 | 0.001818 | [-0.017194, -0.005046] | -0.975857 | -0.035316 | -0.013255 | -0.061872 | 0.004526 | [-0.042206, -0.028884] | -0.918294 |
| avit | MST_06 | 975 | 0.984615 | {'detected->detected': 956, 'detected->valid_no_detection': 3, 'valid_no_detection->detected': 4, 'valid_no_detection->valid_no_detection': 12} | -0.007745 | -0.003362 | -0.014052 | 0.006930 | [-0.013856, -0.001877] | -0.906663 | -0.006248 | -0.001807 | -0.012179 | 0.005386 | [-0.015699, 0.003247] | -0.966335 | -0.006884 | -0.001924 | -0.008279 | 0.004010 | [-0.013007, -0.001023] | -0.889189 | -0.023159 | -0.007176 | -0.059236 | 0.016947 | [-0.030391, -0.016140] | -0.918294 |
| avit | MST_07 | 975 | 0.989744 | {'detected->detected': 959, 'valid_no_detection->detected': 6, 'valid_no_detection->valid_no_detection': 10} | -0.006698 | -0.004555 | -0.018346 | 0.005281 | [-0.012819, -0.000528] | -0.800487 | -0.007994 | -0.003310 | -0.016763 | 0.004196 | [-0.017410, 0.001278] | -0.966335 | -0.004394 | -0.002694 | -0.010756 | 0.003186 | [-0.010566, 0.001696] | -0.889189 | -0.034398 | -0.010501 | -0.073182 | 0.015125 | [-0.042014, -0.026823] | -0.918294 |
| avit | MST_08 | 975 | 0.989744 | {'detected->detected': 956, 'detected->valid_no_detection': 3, 'valid_no_detection->detected': 9, 'valid_no_detection->valid_no_detection': 7} | -0.014831 | -0.015647 | -0.036273 | -0.000949 | [-0.021029, -0.008348] | -0.738980 | -0.025813 | -0.013796 | -0.033577 | 0.000000 | [-0.036147, -0.015919] | -0.833953 | -0.007247 | -0.009401 | -0.021893 | -0.000537 | [-0.013332, -0.000999] | -0.778073 | -0.086698 | -0.044806 | -0.155741 | 0.000000 | [-0.096508, -0.077159] | -0.737597 |
| avit | MST_09 | 975 | 0.995897 | {'detected->detected': 958, 'detected->valid_no_detection': 1, 'valid_no_detection->detected': 13, 'valid_no_detection->valid_no_detection': 3} | -0.025631 | -0.022638 | -0.046494 | -0.005340 | [-0.032665, -0.018441] | -0.923640 | -0.039558 | -0.019831 | -0.044673 | -0.000125 | [-0.050330, -0.028516] | -0.923640 | -0.015823 | -0.013174 | -0.028826 | -0.003142 | [-0.023026, -0.008532] | -0.960304 | -0.119767 | -0.078813 | -0.219950 | -0.001165 | [-0.131056, -0.108643] | -0.941610 |
| avit | MST_10 | 975 | 0.998974 | {'detected->detected': 958, 'detected->valid_no_detection': 1, 'valid_no_detection->detected': 16} | -0.026562 | -0.023662 | -0.052418 | -0.001970 | [-0.034581, -0.018226] | -0.923640 | -0.044903 | -0.021426 | -0.050888 | 0.000000 | [-0.057932, -0.032270] | -0.923640 | -0.014700 | -0.013854 | -0.031987 | -0.001205 | [-0.022771, -0.006402] | -0.960304 | -0.126327 | -0.087948 | -0.232592 | 0.000000 | [-0.139127, -0.113522] | -0.948603 |
| delightsam-dermoscopy | MST_01 | 975 | 0.964103 | {'detected->detected': 929, 'detected->valid_no_detection': 30, 'valid_no_detection->detected': 11, 'valid_no_detection->valid_no_detection': 5} | -0.033805 | -0.007077 | -0.033732 | 0.005443 | [-0.043147, -0.024939] | -0.909291 | -0.042266 | -0.004720 | -0.027137 | 0.003610 | [-0.055528, -0.029188] | -0.959269 | -0.030103 | -0.004061 | -0.020971 | 0.003226 | [-0.040018, -0.020929] | -0.952491 | -0.063705 | -0.031281 | -0.122656 | 0.011382 | [-0.074598, -0.053106] | -0.895164 |
| delightsam-dermoscopy | MST_02 | 975 | 0.972308 | {'detected->detected': 937, 'detected->valid_no_detection': 22, 'valid_no_detection->detected': 11, 'valid_no_detection->valid_no_detection': 5} | -0.030490 | -0.007660 | -0.030787 | 0.003542 | [-0.038936, -0.022270] | -0.908760 | -0.038430 | -0.005835 | -0.027154 | 0.001443 | [-0.051083, -0.025995] | -0.945755 | -0.026347 | -0.004651 | -0.019347 | 0.002136 | [-0.035163, -0.017862] | -0.951430 | -0.065257 | -0.034619 | -0.116551 | 0.007707 | [-0.074881, -0.055628] | -0.916224 |
| delightsam-dermoscopy | MST_03 | 975 | 0.978462 | {'detected->detected': 943, 'detected->valid_no_detection': 16, 'valid_no_detection->detected': 11, 'valid_no_detection->valid_no_detection': 5} | -0.031022 | -0.010585 | -0.034693 | 0.001800 | [-0.039107, -0.023163] | -0.909291 | -0.041648 | -0.007757 | -0.028631 | 0.000096 | [-0.054404, -0.029459] | -0.959269 | -0.025329 | -0.006414 | -0.021603 | 0.001070 | [-0.033716, -0.017265] | -0.952491 | -0.070856 | -0.040758 | -0.130494 | 0.001519 | [-0.080473, -0.061121] | -0.882529 |
| delightsam-dermoscopy | MST_04 | 975 | 0.981538 | {'detected->detected': 950, 'detected->valid_no_detection': 9, 'valid_no_detection->detected': 7, 'valid_no_detection->valid_no_detection': 9} | -0.022234 | -0.007975 | -0.023009 | 0.001711 | [-0.029655, -0.014978] | -0.898506 | -0.031574 | -0.006498 | -0.020691 | 0.000000 | [-0.043061, -0.020320] | -0.945755 | -0.017589 | -0.004741 | -0.014448 | 0.001020 | [-0.025259, -0.010185] | -0.946540 | -0.049367 | -0.028408 | -0.088800 | 0.004164 | [-0.056411, -0.042473] | -0.679425 |
| delightsam-dermoscopy | MST_05 | 975 | 0.979487 | {'detected->detected': 949, 'detected->valid_no_detection': 10, 'valid_no_detection->detected': 6, 'valid_no_detection->valid_no_detection': 10} | -0.014885 | -0.005402 | -0.014193 | 0.002549 | [-0.021772, -0.008351] | -0.887308 | -0.011807 | -0.003988 | -0.013075 | 0.001350 | [-0.021790, -0.001767] | -0.945755 | -0.013292 | -0.003117 | -0.008618 | 0.001580 | [-0.020261, -0.006666] | -0.940290 | -0.032496 | -0.018480 | -0.064932 | 0.005793 | [-0.038366, -0.026645] | -0.717525 |
| delightsam-dermoscopy | MST_06 | 975 | 0.984615 | {'detected->detected': 956, 'detected->valid_no_detection': 3, 'valid_no_detection->detected': 4, 'valid_no_detection->valid_no_detection': 12} | -0.005955 | 0.000191 | -0.010953 | 0.009808 | [-0.012322, 0.000110] | -0.855085 | 0.000115 | 0.000000 | -0.009371 | 0.009157 | [-0.009830, 0.009973] | -0.945755 | -0.006653 | 0.000105 | -0.006452 | 0.005834 | [-0.013119, -0.000547] | -0.877855 | 0.004652 | 0.000000 | -0.037276 | 0.046223 | [-0.002047, 0.011479] | -0.717525 |
| delightsam-dermoscopy | MST_07 | 975 | 0.989744 | {'detected->detected': 959, 'valid_no_detection->detected': 6, 'valid_no_detection->valid_no_detection': 10} | -0.000645 | 0.001455 | -0.011439 | 0.015280 | [-0.006977, 0.005573] | -0.790174 | 0.001717 | 0.000000 | -0.008943 | 0.013503 | [-0.009325, 0.012759] | -0.945755 | -0.001370 | 0.000863 | -0.006855 | 0.009005 | [-0.007700, 0.004764] | -0.805794 | 0.018466 | 0.007008 | -0.035171 | 0.069604 | [0.010991, 0.025943] | -0.717525 |
| delightsam-dermoscopy | MST_08 | 975 | 0.989744 | {'detected->detected': 956, 'detected->valid_no_detection': 3, 'valid_no_detection->detected': 9, 'valid_no_detection->valid_no_detection': 7} | 0.005681 | 0.003820 | -0.011512 | 0.020554 | [-0.000563, 0.012252] | -0.793106 | 0.005050 | 0.002057 | -0.008600 | 0.019117 | [-0.006234, 0.016206] | -0.902562 | 0.004991 | 0.002212 | -0.006935 | 0.012334 | [-0.001144, 0.011490] | -0.805794 | 0.034300 | 0.016342 | -0.031265 | 0.095547 | [0.026333, 0.042252] | -0.540507 |
| delightsam-dermoscopy | MST_09 | 975 | 0.995897 | {'detected->detected': 958, 'detected->valid_no_detection': 1, 'valid_no_detection->detected': 13, 'valid_no_detection->valid_no_detection': 3} | 0.005619 | 0.008010 | -0.012700 | 0.028663 | [-0.001895, 0.013157] | -0.824135 | 0.004203 | 0.005984 | -0.009233 | 0.026645 | [-0.008306, 0.016846] | -0.979296 | 0.003283 | 0.004791 | -0.007669 | 0.016970 | [-0.004176, 0.010714] | -0.894194 | 0.056487 | 0.036109 | -0.026135 | 0.132153 | [0.046309, 0.066471] | -0.566549 |
| delightsam-dermoscopy | MST_10 | 975 | 0.998974 | {'detected->detected': 958, 'detected->valid_no_detection': 1, 'valid_no_detection->detected': 16} | 0.003739 | 0.009437 | -0.020509 | 0.038149 | [-0.005161, 0.012335] | -0.869909 | 0.000381 | 0.005281 | -0.015379 | 0.034955 | [-0.013836, 0.014233] | -0.979296 | 0.002948 | 0.005527 | -0.012389 | 0.022449 | [-0.005681, 0.011410] | -0.924810 | 0.060826 | 0.039298 | -0.041623 | 0.170723 | [0.048074, 0.073522] | -0.727011 |
| vmunet-isic17 | MST_01 | 975 | 0.964103 | {'detected->detected': 929, 'detected->valid_no_detection': 30, 'valid_no_detection->detected': 11, 'valid_no_detection->valid_no_detection': 5} | -0.040355 | -0.006711 | -0.042985 | 0.008684 | [-0.050752, -0.029927] | -0.957943 | -0.057704 | -0.003777 | -0.036033 | 0.005496 | [-0.073904, -0.041440] | -0.971401 | -0.035970 | -0.003727 | -0.026438 | 0.005088 | [-0.046489, -0.025608] | -0.978520 | -0.055639 | -0.010328 | -0.121807 | 0.034444 | [-0.068549, -0.042831] | -0.997831 |
| vmunet-isic17 | MST_02 | 975 | 0.972308 | {'detected->detected': 937, 'detected->valid_no_detection': 22, 'valid_no_detection->detected': 11, 'valid_no_detection->valid_no_detection': 5} | -0.039681 | -0.007310 | -0.041634 | 0.006669 | [-0.049614, -0.030104] | -0.955792 | -0.054891 | -0.004955 | -0.036159 | 0.004145 | [-0.070866, -0.039415] | -0.971401 | -0.034641 | -0.004467 | -0.025418 | 0.003911 | [-0.044317, -0.025228] | -0.974228 | -0.062296 | -0.012751 | -0.123685 | 0.024341 | [-0.074593, -0.050361] | -0.997831 |
| vmunet-isic17 | MST_03 | 975 | 0.978462 | {'detected->detected': 943, 'detected->valid_no_detection': 16, 'valid_no_detection->detected': 11, 'valid_no_detection->valid_no_detection': 5} | -0.039563 | -0.013402 | -0.049983 | 0.004549 | [-0.049376, -0.029984] | -0.895167 | -0.055502 | -0.008289 | -0.043703 | 0.002383 | [-0.071336, -0.040169] | -0.971401 | -0.032498 | -0.007619 | -0.031222 | 0.002551 | [-0.042350, -0.022893] | -0.944684 | -0.073543 | -0.024884 | -0.149794 | 0.018920 | [-0.085882, -0.061368] | -0.988354 |
| vmunet-isic17 | MST_04 | 975 | 0.981538 | {'detected->detected': 950, 'detected->valid_no_detection': 9, 'valid_no_detection->detected': 7, 'valid_no_detection->valid_no_detection': 9} | -0.037811 | -0.012337 | -0.043508 | 0.003647 | [-0.047498, -0.028610] | -0.895167 | -0.051744 | -0.009525 | -0.039921 | 0.001548 | [-0.066189, -0.037655] | -0.971401 | -0.030898 | -0.007171 | -0.027736 | 0.002245 | [-0.040852, -0.021473] | -0.944684 | -0.075903 | -0.029685 | -0.138653 | 0.010388 | [-0.087762, -0.064182] | -0.940128 |
| vmunet-isic17 | MST_05 | 975 | 0.979487 | {'detected->detected': 949, 'detected->valid_no_detection': 10, 'valid_no_detection->detected': 6, 'valid_no_detection->valid_no_detection': 10} | -0.040447 | -0.015588 | -0.047510 | 0.003712 | [-0.049989, -0.031316] | -0.903130 | -0.057564 | -0.012618 | -0.044649 | 0.000703 | [-0.072288, -0.042632] | -0.959949 | -0.033035 | -0.009475 | -0.028865 | 0.001997 | [-0.042685, -0.023821] | -0.935612 | -0.077094 | -0.034812 | -0.152230 | 0.010869 | [-0.088669, -0.065501] | -0.923807 |
| vmunet-isic17 | MST_06 | 975 | 0.984615 | {'detected->detected': 956, 'detected->valid_no_detection': 3, 'valid_no_detection->detected': 4, 'valid_no_detection->valid_no_detection': 12} | -0.148655 | -0.065028 | -0.195149 | -0.011552 | [-0.163792, -0.134061] | -0.970796 | -0.224608 | -0.054518 | -0.250698 | -0.003038 | [-0.248257, -0.201400] | -0.980877 | -0.122562 | -0.040606 | -0.129194 | -0.007311 | [-0.137147, -0.108344] | -0.984284 | -0.167150 | -0.107329 | -0.324925 | 0.000000 | [-0.183438, -0.150618] | -1.000000 |
| vmunet-isic17 | MST_07 | 975 | 0.989744 | {'detected->detected': 959, 'valid_no_detection->detected': 6, 'valid_no_detection->valid_no_detection': 10} | -0.137717 | -0.062996 | -0.170713 | -0.007876 | [-0.153115, -0.122971] | -0.969279 | -0.214829 | -0.056723 | -0.210229 | -0.000902 | [-0.238054, -0.192214] | -0.980877 | -0.112267 | -0.038121 | -0.108991 | -0.005033 | [-0.127299, -0.097981] | -0.981259 | -0.162476 | -0.102495 | -0.310517 | 0.003224 | [-0.179013, -0.145944] | -1.000000 |
| vmunet-isic17 | MST_08 | 975 | 0.989744 | {'detected->detected': 956, 'detected->valid_no_detection': 3, 'valid_no_detection->detected': 9, 'valid_no_detection->valid_no_detection': 7} | -0.151778 | -0.057325 | -0.170777 | -0.007744 | [-0.169755, -0.134439] | -0.977709 | -0.213033 | -0.052952 | -0.233080 | -0.000141 | [-0.237784, -0.189022] | -0.980877 | -0.131652 | -0.034970 | -0.112016 | -0.004427 | [-0.149784, -0.114542] | -0.988729 | -0.157149 | -0.104547 | -0.285224 | 0.013066 | [-0.174016, -0.140427] | -1.000000 |
| vmunet-isic17 | MST_09 | 975 | 0.995897 | {'detected->detected': 958, 'detected->valid_no_detection': 1, 'valid_no_detection->detected': 13, 'valid_no_detection->valid_no_detection': 3} | -0.260425 | -0.111098 | -0.472362 | -0.026222 | [-0.281997, -0.239121] | -0.980877 | -0.346870 | -0.109909 | -0.830262 | -0.017595 | [-0.373117, -0.319440] | -0.980877 | -0.239934 | -0.071698 | -0.380436 | -0.016416 | [-0.262430, -0.217858] | -0.990346 | -0.234469 | -0.176960 | -0.421148 | -0.010377 | [-0.252979, -0.215662] | -1.000000 |
| vmunet-isic17 | MST_10 | 975 | 0.998974 | {'detected->detected': 958, 'detected->valid_no_detection': 1, 'valid_no_detection->detected': 16} | -0.340518 | -0.194285 | -0.724144 | -0.045565 | [-0.364386, -0.317536] | -0.980877 | -0.437189 | -0.234864 | -0.869346 | -0.028020 | [-0.464470, -0.410259] | -0.980877 | -0.323244 | -0.131615 | -0.739456 | -0.028606 | [-0.348761, -0.298898] | -0.990346 | -0.269811 | -0.200048 | -0.504962 | -0.010637 | [-0.290808, -0.249005] | -1.000000 |

### MSKCC: resultados globales y todos los estratos

Los estratos `insufficient` permanecen explícitamente `NA`; no se fusionaron ni imputaron. Los IC95% se omiten de esta tabla ancha solo cuando el estrato es insuficiente; para estratos completos se conserva el estimador puntual y las métricas de error.

| Método | Familia | Estrato | Estado | N | Pacientes | ICC | Bias | MAE | RMSE | LoA |
|---|---|---|---|---|---|---|---|---|---|---|
| avit | global | all | complete | 1168 | 46 | 0.335650 | -4.089053 | 60.278374 | 75.767751 | [-152.44094214157275, 144.26283677125957] |
| avit | anatom_site_general | anterior torso | complete | 225 | 46 | 0.504484 | 9.304177 | 47.237309 | 57.157997 | [-101.47774579102379, 120.08610041746196] |
| avit | anatom_site_general | head/neck | complete | 116 | 43 | 0.251703 | -17.570242 | 59.777881 | 78.554199 | [-168.2867619714071, 133.14627788203484] |
| avit | anatom_site_general | lateral torso | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| avit | anatom_site_general | lower extremity | complete | 373 | 45 | 0.233718 | -12.863015 | 76.160955 | 91.305168 | [-190.27432972688842, 164.5483002144557] |
| avit | anatom_site_general | palms/soles | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| avit | anatom_site_general | posterior torso | complete | 169 | 42 | 0.589632 | 10.092802 | 43.504403 | 52.135154 | [-90.45696145454382, 110.64256595693799] |
| avit | anatom_site_general | upper extremity | complete | 285 | 44 | 0.307817 | -6.102051 | 59.937654 | 76.900206 | [-156.61548038200993, 144.41137911227514] |
| avit | anatomic_site | abdomen | complete | 103 | 44 | 0.536267 | 18.995294 | 48.864646 | 59.622650 | [-92.31740534508965, 130.30799342469942] |
| avit | anatomic_site | dorsal forearm | complete | 107 | 42 | 0.399216 | 1.591776 | 50.799747 | 65.270264 | [-126.90173930310537, 130.0852906490863] |
| avit | anatomic_site | head/neck | complete | 116 | 43 | 0.251703 | -17.570242 | 59.777881 | 78.554199 | [-168.2867619714071, 133.14627788203484] |
| avit | anatomic_site | lateral torso | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| avit | anatomic_site | lower back | complete | 85 | 39 | 0.680278 | 23.307235 | 41.854876 | 47.472816 | [-58.23450004590611, 104.84897057814123] |
| avit | anatomic_site | lower leg | complete | 235 | 45 | 0.372903 | -0.231032 | 63.363130 | 79.182002 | [-155.75835663558544, 155.29629286075712] |
| avit | anatomic_site | palms/soles | complete | 138 | 40 | 0.059419 | -34.374000 | 97.954353 | 108.886454 | [-237.61575059537168, 168.8677504428155] |
| avit | anatomic_site | upper arm | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| avit | anatomic_site | upper back | complete | 84 | 40 | 0.489622 | -3.278945 | 45.173566 | 56.462456 | [-114.4221395928785, 107.86424870793363] |
| avit | anatomic_site | upper chest | complete | 122 | 46 | 0.444072 | 1.122333 | 45.863410 | 54.991253 | [-107.08244837228797, 109.32711409973705] |
| avit | anatomic_site | upper leg | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| avit | anatomic_site | ventral forearm | complete | 178 | 42 | 0.260548 | -10.726991 | 65.430665 | 83.111732 | [-172.71915188557796, 151.26516954909266] |
| avit | dermoscopic_type | contact non-polarized | complete | 278 | 43 | 0.217709 | -35.312764 | 78.619201 | 94.021655 | [-206.41180648510493, 135.78627765595996] |
| avit | dermoscopic_type | contact polarized | complete | 241 | 45 | 0.345607 | 2.001537 | 46.350458 | 62.685923 | [-121.05579560659694, 125.0588705762149] |
| avit | dermoscopic_type | non-contact polarized | complete | 252 | 45 | 0.448571 | 16.199620 | 44.736818 | 58.741629 | [-94.68950029989963, 127.08874102623477] |
| avit | image_type | clinical: close-up | complete | 397 | 46 | 0.395121 | 1.199648 | 65.755325 | 78.253885 | [-152.35345637317567, 154.75275161564952] |
| avit | image_type | dermoscopic | complete | 771 | 46 | 0.297649 | -6.812287 | 57.458206 | 74.455231 | [-152.22676460816407, 138.60218966161685] |
| delightsam-dermoscopy | global | all | complete | 699 | 46 | 0.448968 | 14.754514 | 47.964916 | 61.362659 | [-102.07139837868696, 131.58042701514395] |
| delightsam-dermoscopy | anatom_site_general | anterior torso | complete | 142 | 41 | 0.538965 | 12.602613 | 40.446691 | 50.015761 | [-82.60106140650022, 107.80628731983133] |
| delightsam-dermoscopy | anatom_site_general | head/neck | complete | 68 | 32 | 0.333158 | -1.954680 | 48.889345 | 67.012194 | [-134.21882475803068, 130.3094649315193] |
| delightsam-dermoscopy | anatom_site_general | lateral torso | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| delightsam-dermoscopy | anatom_site_general | lower extremity | complete | 249 | 44 | 0.395520 | 21.224294 | 57.992946 | 72.061420 | [-114.02280792343245, 156.47139676674388] |
| delightsam-dermoscopy | anatom_site_general | palms/soles | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| delightsam-dermoscopy | anatom_site_general | posterior torso | complete | 99 | 38 | 0.660630 | 20.869601 | 37.536484 | 42.488011 | [-52.03784819758805, 93.77704990324192] |
| delightsam-dermoscopy | anatom_site_general | upper extremity | complete | 141 | 38 | 0.426008 | 9.261084 | 44.703655 | 59.529701 | [-106.40744307724259, 124.92961197475267] |
| delightsam-dermoscopy | anatomic_site | abdomen | complete | 52 | 29 | 0.554350 | 14.161866 | 43.882618 | 54.397149 | [-89.78429951060907, 118.10803178741317] |
| delightsam-dermoscopy | anatomic_site | dorsal forearm | complete | 61 | 30 | 0.344688 | 1.688558 | 45.135566 | 60.840702 | [-118.50252595440504, 121.87964152765558] |
| delightsam-dermoscopy | anatomic_site | head/neck | complete | 68 | 32 | 0.333158 | -1.954680 | 48.889345 | 67.012194 | [-134.21882475803068, 130.3094649315193] |
| delightsam-dermoscopy | anatomic_site | lateral torso | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| delightsam-dermoscopy | anatomic_site | lower back | complete | 45 | 24 | 0.634094 | 27.998798 | 40.192293 | 45.586114 | [-43.307872752928574, 99.30546864195247] |
| delightsam-dermoscopy | anatomic_site | lower leg | complete | 170 | 44 | 0.467331 | 16.441610 | 50.853299 | 64.789693 | [-106.75209526893386, 139.6353160745933] |
| delightsam-dermoscopy | anatomic_site | palms/soles | complete | 79 | 32 | 0.273887 | 31.516146 | 73.356742 | 85.640512 | [-125.55709714408438, 188.58938935955837] |
| delightsam-dermoscopy | anatomic_site | upper arm | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| delightsam-dermoscopy | anatomic_site | upper back | complete | 54 | 33 | 0.671603 | 14.928603 | 35.323310 | 39.722103 | [-57.89661938832222, 87.75382594116772] |
| delightsam-dermoscopy | anatomic_site | upper chest | complete | 90 | 37 | 0.525943 | 11.701711 | 38.461488 | 47.299666 | [-78.62702183784083, 102.03044407449863] |
| delightsam-dermoscopy | anatomic_site | upper leg | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| delightsam-dermoscopy | anatomic_site | ventral forearm | complete | 80 | 33 | 0.455115 | 15.035136 | 44.374323 | 58.510327 | [-96.49345408714547, 126.56372614440343] |
| delightsam-dermoscopy | dermoscopic_type | contact non-polarized | complete | 179 | 44 | 0.335625 | -0.047926 | 54.724135 | 71.209746 | [-140.0105014097901, 139.914648566908] |
| delightsam-dermoscopy | dermoscopic_type | contact polarized | complete | 146 | 42 | 0.529592 | 22.726658 | 35.466281 | 44.562550 | [-52.662114894229305, 98.1154300400119] |
| delightsam-dermoscopy | dermoscopic_type | non-contact polarized | complete | 172 | 43 | 0.583192 | 27.750313 | 36.582810 | 45.181650 | [-42.338152937821974, 97.83877809761816] |
| delightsam-dermoscopy | image_type | clinical: close-up | complete | 202 | 44 | 0.438030 | 11.043755 | 60.700672 | 73.230685 | [-131.19934990842805, 153.28686048758712] |
| delightsam-dermoscopy | image_type | dermoscopic | complete | 497 | 45 | 0.457235 | 16.262710 | 42.788613 | 55.822471 | [-88.50881097585705, 121.03423125733252] |
| vmunet-isic17 | global | all | complete | 1396 | 46 | 0.514475 | 11.515923 | 47.652770 | 59.835706 | [-103.61079190807934, 126.6426384962243] |
| vmunet-isic17 | anatom_site_general | anterior torso | complete | 281 | 46 | 0.619521 | 12.384978 | 41.034497 | 48.130281 | [-78.9363369823644, 103.70629214430068] |
| vmunet-isic17 | anatom_site_general | head/neck | complete | 109 | 40 | 0.278340 | -11.662832 | 55.127351 | 72.253240 | [-152.0676308433726, 128.74196683172644] |
| vmunet-isic17 | anatom_site_general | lateral torso | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| vmunet-isic17 | anatom_site_general | lower extremity | complete | 397 | 45 | 0.407479 | 14.807890 | 60.830163 | 75.570505 | [-130.62220005894397, 160.23798031630184] |
| vmunet-isic17 | anatom_site_general | palms/soles | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| vmunet-isic17 | anatom_site_general | posterior torso | complete | 248 | 43 | 0.719559 | 13.569123 | 36.334571 | 42.059505 | [-64.61737207939764, 91.75561761911347] |
| vmunet-isic17 | anatom_site_general | upper extremity | complete | 361 | 44 | 0.550841 | 12.807269 | 43.831422 | 54.498313 | [-91.1620864034439, 116.77662423667137] |
| vmunet-isic17 | anatomic_site | abdomen | complete | 128 | 43 | 0.672536 | 17.375577 | 41.469203 | 47.422212 | [-69.44790635814397, 104.19906105061312] |
| vmunet-isic17 | anatomic_site | dorsal forearm | complete | 133 | 40 | 0.533220 | 9.951103 | 42.767267 | 54.608024 | [-95.68640292229539, 115.58860835411924] |
| vmunet-isic17 | anatomic_site | head/neck | complete | 109 | 40 | 0.278340 | -11.662832 | 55.127351 | 72.253240 | [-152.0676308433726, 128.74196683172644] |
| vmunet-isic17 | anatomic_site | lateral torso | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| vmunet-isic17 | anatomic_site | lower back | complete | 121 | 39 | 0.720099 | 16.779978 | 37.300883 | 43.564279 | [-62.34548446471318, 95.90544076205335] |
| vmunet-isic17 | anatomic_site | lower leg | complete | 287 | 44 | 0.457434 | 10.530906 | 54.263379 | 70.005540 | [-125.35553485413448, 146.41734590091625] |
| vmunet-isic17 | anatomic_site | palms/soles | complete | 110 | 38 | 0.289023 | 25.966932 | 77.963499 | 88.456728 | [-140.52823722101067, 192.46210078232616] |
| vmunet-isic17 | anatomic_site | upper arm | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| vmunet-isic17 | anatomic_site | upper back | complete | 127 | 43 | 0.715283 | 10.509961 | 35.413911 | 40.573940 | [-66.60487086520268, 87.62479355710317] |
| vmunet-isic17 | anatomic_site | upper chest | complete | 153 | 46 | 0.560986 | 8.209835 | 40.670821 | 48.714746 | [-86.21446089108039, 102.63413147845321] |
| vmunet-isic17 | anatomic_site | upper leg | insufficient | 0 | NA | NA | NA | NA | NA | NA |
| vmunet-isic17 | anatomic_site | ventral forearm | complete | 228 | 42 | 0.552409 | 14.473366 | 44.452180 | 54.434213 | [-88.60354785387301, 117.55027958791928] |
| vmunet-isic17 | dermoscopic_type | contact non-polarized | complete | 352 | 45 | 0.480441 | -1.920008 | 53.597254 | 66.713952 | [-132.81124799544452, 128.9712326813732] |
| vmunet-isic17 | dermoscopic_type | contact polarized | complete | 306 | 46 | 0.642129 | 19.498670 | 34.402722 | 42.772005 | [-55.238726415169374, 94.23606616796005] |
| vmunet-isic17 | dermoscopic_type | non-contact polarized | complete | 352 | 46 | 0.634323 | 22.064952 | 34.728852 | 43.919268 | [-52.4704330098823, 96.60033780965867] |
| vmunet-isic17 | image_type | clinical: close-up | complete | 386 | 46 | 0.437022 | 7.820247 | 64.521358 | 75.368992 | [-139.29631759195522, 154.93681182441182] |
| vmunet-isic17 | image_type | dermoscopic | complete | 1010 | 46 | 0.562097 | 12.928330 | 41.205963 | 52.703618 | [-87.26423293502012, 113.1208933937549] |

### Resumen destacado de cada etapa y rama

| Etapa/rama | Resultado destacado | Valor que se conservó o conclusión permitida |
|---|---|---|
| Datos | 2594 Training / 100 Validation / 1000 Test | Folds OOF solo en Training; selección solo en Validation; Test no reordena modelos |
| YOLO CV | Mejor mAP@0.5:0.95: fold 1 = 0.7157 | Todos los folds completaron 300 epochs; `best.pt` por fitness oficial |
| ROI | 2551/2594 detectadas; 43 no-detecciones | Margen $m^*=0.1225400188$; fallback a imagen completa |
| Roster | 16/16 con 100 resultados científicos | Ningún checkpoint borrado; predicciones degeneradas se conservaron |
| Ranking | AViT TJ=0.778707 | TOP-3 = AViT, DeLightSAM, VM-UNet ISIC17 |
| Incertidumbre | AViT $P(\text{rank}=1)=0.5624$ | El TOP-3 es selección operacional; no superioridad universal [21], [22] |
| Comparaciones | AViT−DeLightSAM ΔTJ=0.011365, $p_{Holm}=1$ | No se afirma diferencia estadística entre los tres primeros |
| D35 | 300 pares método–imagen calibraron el margen | $r^*=0.1418421924$ del lado menor de ROI |
| Color Validation | 299/300 disponibles | 1 VM-UNet `unavailable_insufficient_clean_skin`; no imputación |
| Freeze | Commit `1c6ddbe8838d0734e08cb5eaa5d009219212f28c` | Apertura única posterior; mismos hashes en reanudaciones |
| Test original | AViT TJ=0.766669 y J=0.809185 | AViT mantuvo TOP-1; orden congelado no se modificó |
| Robustez ITA | VM-UNet mostró asociación positiva pequeña | Asociación con color observado, no fairness demográfica [23], [31] |
| Síntesis MST | 975 fuentes completas; 25 no disponibles | 9750 PNG lossless; 250 condiciones `unavailable` |
| Detector MST | Tasa 0.964103–0.998974 según tono | YOLO se reejecutó por condición; fallback incluido |
| AViT MST | Peor media: MST10 ΔJ=−0.026562 | Degradación moderada, persistente |
| DeLightSAM MST | MST07–10 alrededor de cero | Mayor estabilidad media en tonos finales; IC incluyen cero |
| VM-UNet MST | MST10 ΔJ=−0.340518 | Fragilidad cromática marcada desde MST06 |
| MSKCC global | VM-UNet ICC=0.514475, MAE=47.652770 | Mejor de los tres, pero acuerdo absoluto todavía limitado |
| MSKCC estratos | Varios estratos sin observaciones legítimas | Se reportan `NA`; no se agrupan ni imputan |
| Cierre | Job 25037 emitió `V2_COMPLETE` | G1–G8 y procedencia final verificados |

### Matriz de lectura y uso de las 37 referencias canónicas

Las 37 fuentes fueron contrastadas nuevamente con su fuente editorial, repositorio oficial, DOI o registro primario. La última columna evita sobreextender la evidencia.

| Ref. | Uso científico en V2 | Límite de la cita |
|---|---|---|
| 1 | ISIC 2018: particiones y métrica TJ | No fija hiperparámetros V2 |
| 2 | Arquitectura YOLOv7 | No valida dermatoscopia por sí sola |
| 3 | Uso de CNN/detección en lesiones | No prescribe esta configuración |
| 4 | ROI aproximada en dermatoscopia | No prescribe Q95 |
| 5 | Detector seguido de segmentación | No fija fallback |
| 6 | Pelo como artefacto; DullRazor | No autoriza inpainting colorimétrico |
| 7 | Pelo/inpainting puede afectar segmentación | No se usa su red |
| 8 | Hair removal profundo y máscaras | No se usa su GAN |
| 9 | Preprocesamiento depende del método | No justifica alterar entradas aquí |
| 10 | Pelo claro/oscuro y marcas | No se replica SharpRazor |
| 11 | Anotación fina de pelo | No entrena el detector V2 |
| 12 | Dataset asociado de pelo | No se mezcla con ISIC |
| 13 | Código y fitness YOLOv7 | No reemplaza el commit fijado |
| 14 | Marco YOLO one-stage | No aporta resultados V2 |
| 15 | Transferencia general→específica | No prueba ausencia de leakage |
| 16 | Fine tuning en imagen médica | No prescribe 300 epochs |
| 17 | Límites de transferencia médica | No selecciona el modelo |
| 18 | Protocolo oficial Task 1 | No permite abrir Test antes del freeze |
| 19 | Transparencia/reproducibilidad CLAIM | No es una métrica |
| 20 | Definición e interpretación de métricas | No decide TJ primaria |
| 21 | Inestabilidad de rankings | No altera regla predeclarada |
| 22 | Bootstrap/análisis de challenges | No convierte ranking en causalidad |
| 23 | Sesgo por tono en segmentación | No demuestra demografía en ISIC V2 |
| 24 | Medición reproducible de color de piel | No equipara JPEG con colorímetro |
| 25 | Fiabilidad depende de protocolo | No valida automáticamente esta cámara |
| 26 | Detección de reflejos | No replica el algoritmo endoscópico |
| 27 | ITA y pigmentación | No autoriza inferir raza/Fitzpatrick |
| 28 | Definición sRGB | No garantiza calibración del sensor |
| 29 | XYZ/CIELAB/colorimetría CIE | No fija paleta MST |
| 30 | Contexto sociocultural del aclaramiento | No aporta algoritmo |
| 31 | Transparencia de datasets y bias | No prueba fairness clínica |
| 32 | CIEDE2000 | No es la variable primaria MST |
| 33 | Comparación MST/FST/colorímetro | No permite categorizar ITA automáticamente |
| 34 | CLAIM 2024; estándar de referencia y particiones | No sustituye validación externa |
| 35 | Bias y límites de acuerdo | Correlación no es acuerdo |
| 36 | Concordancia reproducible | No es el ICC implementado |
| 37 | Acuerdo ordinal ponderado | Kappa quedó planificado/no ejecutado |

### Cobertura completa de la bitácora de ejecución 19.1–19.189

La bitácora canónica es append-only. Se agrupa aquí por fase para no confundir cientos de observaciones operativas con resultados científicos, pero la columna final enumera los 189 encabezados sin omitir intentos fallidos.

| Rango | Fase | Eventos incluidos |
|---|---|---|
| 19.1–19.24 | arranque, entorno y dependencias | 19.1 Preflight inicial del entorno V2; 19.2 Resolución de remoto y conectividad CEDIA; 19.3 Aprobación de concordancia MSKCC; 19.4 Aprobación de robustez real y límites de fairness; 19.5 Aprobación del `scientific freeze`; 19.6 Aprobación del protocolo de Test sellado; 19.7 Aprobación del roster V2; 19.8 Aprobación del runbook CEDIA; 19.9 Implementación del núcleo reproducible V2; 19.10 Despliegue inicial y transferencia no destructiva a CEDIA; 19.11 Validación estática inicial de etapas Slurm; 19.12 Pipeline de desarrollo, selección y colorimetría publicado; 19.13 Finalización de transferencia y corrección de ejecutabilidad; 19.14 Primer preflight científico en CEDIA; 19.15 Disponibilidad oficial de GT Test y MSKCC; 19.16 Conversión inversa CIELAB para síntesis MST; 19.17 Primer intento del orquestador completo; 19.18 Bootstrap CEDIA sin `ensurepip`; 19.19 Ausencia de DNS en nodos compute y wheelhouse offline; 19.20 Dependencias transitivas ausentes en el wheelhouse inicial; 19.21 Restricción de versión de `huggingface-hub`; 19.22 Metadato residual en instalación `pip --target`; 19.23 Preparación Development y dependencia de ejecución YOLOv7; 19.24 Longitud de socket `AF_UNIX` en DataLoader YOLOv7 |
| 19.25–19.60 | monitoreo fold 0 | 19.25 Primer seguimiento prolongado de YOLOv7 CV corregido; 19.26 Segundo seguimiento prolongado de YOLOv7 CV; 19.27 Tercer seguimiento prolongado de YOLOv7 CV; 19.28 Cuarto seguimiento prolongado de YOLOv7 CV; 19.29 Quinto seguimiento prolongado de YOLOv7 CV; 19.30 Sexto seguimiento prolongado de YOLOv7 CV; 19.31 Séptimo seguimiento prolongado de YOLOv7 CV; 19.32 Octavo seguimiento prolongado de YOLOv7 CV; 19.33 Noveno seguimiento prolongado de YOLOv7 CV; 19.34 Décimo seguimiento prolongado de YOLOv7 CV; 19.35 Undécimo seguimiento prolongado de YOLOv7 CV; 19.36 Duodécimo seguimiento prolongado de YOLOv7 CV; 19.37 Decimotercer seguimiento prolongado de YOLOv7 CV; 19.38 Decimocuarto seguimiento prolongado de YOLOv7 CV; 19.39 Decimoquinto seguimiento prolongado de YOLOv7 CV; 19.40 Decimosexto seguimiento prolongado de YOLOv7 CV; 19.41 Decimoséptimo seguimiento prolongado de YOLOv7 CV; 19.42 Decimoctavo seguimiento prolongado de YOLOv7 CV; 19.43 Decimonoveno seguimiento prolongado de YOLOv7 CV; 19.44 Vigésimo seguimiento prolongado de YOLOv7 CV; 19.45 Vigesimoprimer seguimiento prolongado de YOLOv7 CV; 19.46 Vigesimosegundo seguimiento prolongado de YOLOv7 CV; 19.47 Vigesimotercer seguimiento prolongado de YOLOv7 CV; 19.48 Vigesimocuarto seguimiento prolongado de YOLOv7 CV; 19.49 Vigesimoquinto seguimiento prolongado de YOLOv7 CV; 19.50 Vigesimosexto seguimiento prolongado de YOLOv7 CV; 19.51 Vigesimoséptimo seguimiento prolongado de YOLOv7 CV; 19.52 Vigesimoctavo seguimiento prolongado de YOLOv7 CV; 19.53 Vigesimonoveno seguimiento prolongado de YOLOv7 CV; 19.54 Trigésimo seguimiento prolongado de YOLOv7 CV; 19.55 Trigésimo primer seguimiento prolongado de YOLOv7 CV; 19.56 Trigésimo segundo seguimiento prolongado de YOLOv7 CV; 19.57 Trigésimo tercer seguimiento prolongado de YOLOv7 CV; 19.58 Trigésimo cuarto seguimiento prolongado de YOLOv7 CV; 19.59 Trigésimo quinto seguimiento prolongado de YOLOv7 CV; 19.60 Trigésimo sexto seguimiento prolongado de YOLOv7 CV |
| 19.61–19.82 | fold 1 y recovery | 19.61 Finalización del fold 0 y espera de liberación del fold 1; 19.62 Inicio automático y verificación temprana del fold 1; 19.63 Segundo seguimiento del fold 1 de YOLOv7 CV; 19.64 Tercer seguimiento del fold 1 de YOLOv7 CV; 19.65 Cuarto seguimiento del fold 1 de YOLOv7 CV; 19.66 Quinto seguimiento del fold 1 de YOLOv7 CV; 19.67 Sexto seguimiento del fold 1 de YOLOv7 CV; 19.68 Séptimo seguimiento del fold 1 de YOLOv7 CV; 19.69 Octavo seguimiento del fold 1 de YOLOv7 CV; 19.70 Noveno seguimiento del fold 1 y verificación de escritura de checkpoint; 19.71 Décimo seguimiento del fold 1 de YOLOv7 CV; 19.72 Undécimo seguimiento del fold 1 de YOLOv7 CV; 19.73 Duodécimo seguimiento del fold 1 de YOLOv7 CV; 19.74 Decimotercer seguimiento del fold 1 de YOLOv7 CV; 19.75 Decimocuarto seguimiento del fold 1 de YOLOv7 CV; 19.76 Decimoquinto seguimiento del fold 1 de YOLOv7 CV; 19.77 Decimosexto seguimiento del fold 1 de YOLOv7 CV; 19.78 Decimoséptimo seguimiento del fold 1 de YOLOv7 CV; 19.79 Decimoctavo seguimiento del fold 1 de YOLOv7 CV; 19.80 Decimonoveno seguimiento del fold 1 de YOLOv7 CV; 19.81 Vigésimo seguimiento del fold 1 de YOLOv7 CV; 19.82 Fallo del supervisor del fold 1 y recuperación trazable desde checkpoint |
| 19.83–19.131 | folds 2–4, recoveries y OOF | 19.83 Verificación posterior a la reparación y seguimiento temprano del fold 2; 19.84 Segundo seguimiento del fold 2 tras la reparación del DAG; 19.85 Tercer seguimiento del fold 2 tras la reparación del DAG; 19.86 Cuarto seguimiento del fold 2 tras la reparación del DAG; 19.87 Segundo SIGKILL externo y recuperación encadenada del fold 2; 19.88 Seguimiento temprano del fold 3 y cadena doble de recuperación; 19.89 Segundo seguimiento del fold 3 y cadena doble de recuperación; 19.90 Tercer seguimiento del fold 3 y cadena doble de recuperación; 19.91 Cuarto seguimiento del fold 3 y cadena doble de recuperación; 19.92 Quinto seguimiento del fold 3 y cadena doble de recuperación; 19.93 Sexto seguimiento del fold 3 y cadena doble de recuperación; 19.94 Séptimo seguimiento del fold 3 y corrección de consulta del ledger; 19.95 Octavo seguimiento del fold 3 y reintento de autorización SSH; 19.96 Noveno seguimiento del fold 3 y cadena doble de recuperación; 19.97 Décimo seguimiento del fold 3 y cadena doble de recuperación; 19.98 Undécimo seguimiento del fold 3 y segundo reintento de autorización SSH; 19.99 Duodécimo seguimiento del fold 3 y cadena doble de recuperación; 19.100 Decimotercer seguimiento del fold 3 y cadena doble de recuperación; 19.101 Decimocuarto seguimiento del fold 3 y cadena doble de recuperación; 19.102 Decimoquinto seguimiento del fold 3 y cadena doble de recuperación; 19.103 Decimosexto seguimiento del fold 3 y cadena doble de recuperación; 19.104 Decimoséptimo seguimiento del fold 3 y cadena doble de recuperación; 19.105 Decimoctavo seguimiento del fold 3 y cadena doble de recuperación; 19.106 Decimonoveno seguimiento del fold 3 y cadena doble de recuperación; 19.107 Vigésimo seguimiento del fold 3 y cadena doble de recuperación; 19.108 Vigesimoprimer seguimiento del fold 3 y cadena doble de recuperación; 19.109 Vigesimosegundo seguimiento del fold 3 y cadena doble de recuperación; 19.110 Vigesimotercer seguimiento del fold 3 y cadena doble de recuperación; 19.111 Vigesimocuarto seguimiento del fold 3 y cadena doble de recuperación; 19.112 Vigesimoquinto seguimiento del fold 3 y cadena doble de recuperación; 19.113 Vigesimosexto seguimiento del fold 3 y cadena doble de recuperación; 19.114 Vigesimoséptimo seguimiento del fold 3 y cadena doble de recuperación; 19.115 Vigesimoctavo seguimiento del fold 3 y cadena doble de recuperación; 19.116 Vigesimonoveno seguimiento del fold 3 y cadena doble de recuperación; 19.117 Finalización exitosa del fold 3 y espera de liberación del array; 19.118 Arranque automático y primer seguimiento del fold 4; 19.119 Segundo seguimiento del fold 4 y cadena doble de recuperación; 19.120 Seguimiento consolidado del fold 4 durante el intervalo nocturno; 19.121 Cuarto seguimiento del fold 4 y cadena doble de recuperación; 19.122 Quinto seguimiento del fold 4 y cadena doble de recuperación; 19.123 Cierre del fold 4 y seguimiento consolidado del recovery fold 2; 19.124 Segundo seguimiento del recovery fold 2; 19.125 Tercer seguimiento del recovery fold 2; 19.126 Cuarto seguimiento del recovery fold 2; 19.127 Finalización del recovery fold 2 y arranque del recovery fold 1; 19.128 Segundo seguimiento del recovery fold 1; 19.129 Tercer seguimiento del recovery fold 1; 19.130 Cuarto seguimiento del recovery fold 1; 19.131 Cierre del recovery fold 1, OOF completo, calibración D09 y arranque del entrenamiento final |
| 19.132–19.145 | entrenamiento final, Validation y TOP-3 | 19.132 SIGKILL del entrenamiento final y reparación conservadora del DAG; 19.133 Segundo SIGKILL del entrenamiento final y nueva reanudación idempotente; 19.134 Seguimiento del segundo recovery del entrenamiento final; 19.135 Tercer SIGKILL del entrenamiento final y continuidad desde época 102; 19.136 Seguimiento del tercer recovery del entrenamiento final; 19.137 Continuidad prolongada del tercer recovery; 19.138 Cuarto seguimiento del tercer recovery del entrenamiento final; 19.139 Quinto seguimiento del tercer recovery del entrenamiento final; 19.140 Sexto seguimiento del tercer recovery del entrenamiento final; 19.141 Tramo terminal del tercer recovery del entrenamiento final; 19.142 Entrenamiento final y YOLO Validation completos; segmentadores en ejecución; 19.143 Quince checkpoints completos y GrabCut activo en Validation; 19.144 GrabCut alcanza 85/100 y se ratifica la separación Validation/Test; 19.145 Validation completa, TOP-3 V2 y reparación de la paleta MST pre-freeze |
| 19.146–19.158 | freeze y Test original | 19.146 Scientific freeze verificado, publicado y activación controlada del Test; 19.147 Fallo previo a la copia de Test y recuperación operativa; 19.148 Apertura Test completada, inferencia YOLO validada y segmentación TOP-3 iniciada; 19.149 Progreso saludable de AViT sobre Test; 19.150 AViT Test completo e inicio automático de DelightSAM; 19.151 Seguimiento saludable de DelightSAM sobre Test; 19.152 DelightSAM alcanza 851/1000 sin incidencias; 19.153 DelightSAM alcanza 944/1000 y mantiene salud operativa; 19.154 DelightSAM Test completo e inicio de VM-UNet ISIC17; 19.155 VM-UNet ISIC17 alcanza 528/1000 en Test; 19.156 VM-UNet ISIC17 alcanza 840/1000 en Test; 19.157 TOP-3 Test completo e inicio del análisis original; 19.158 Análisis Test original completado |
| 19.159–19.173 | MST inicial, fallos D45 y rediseño D59–D60 | 19.159 Generación MST activa sobre las 1000 fuentes Test; 19.160 Seguimiento de generación MST: 572 variantes; 19.161 Generación MST alcanza 690/10000; 19.162 Generación MST alcanza 815 variantes; 19.163 Fallo D45 y detención por invalidez del código científico congelado; 19.164 Aprobación y controles de la corrección D37/D45; 19.165 Publicación, validación CEDIA y reinicio desde cero de MST; 19.166 Detención total solicitada para rediseñar el flujo MST/YOLO; 19.167 Aprobación del rediseño MST end-to-end sobre imagen completa; 19.168 Implementación local del rediseño MST end-to-end; 19.169 Publicación, validación CEDIA y limpieza D59; 19.170 Fallo de orquestación 24326 y reinicio aislado 24336; 19.171 Detención solicitada de 24336 y aprobación de reanudación paralela D60; 19.172 Validación D60, colisión 24349 y relanzamiento 24359; 19.173 Corrección de paralelismo efectivo y reanudación 24476 |
| 19.174–19.186 | MST reanudado, MSKCC y cierre | 19.174 Progreso verificado de la generación MST 24476; 19.175 Progreso mayoritario de MST 24476; 19.176 Generación MST completa e inicio de YOLOv7 por condición; 19.177 YOLOv7 MST completo y segmentación iniciada; 19.178 Fallo externo AViT 24478_0 y reanudación verificable; 19.179 Relanzamiento segmentación MST 24521–24529; 19.180 Segundo SIGKILL AViT y preservación efectiva del checkpoint; 19.181 DeLightSAM MST completo e inicio de VM-UNet; 19.182 Cierre de VM-UNet MST y diagnóstico del fallo de AViT; 19.183 Reanudación de AViT MST y reconstrucción de la cadena afterok; 19.184 Cierre de MST y fallo reproducible en la entrada YOLO de MSKCC; 19.185 Corrección MSKCC sin ground truth y relanzamiento desde la etapa fallida; 19.186 Finalización end-to-end de la metodología V2 |
| 19.187–19.189 | documentación | 19.187 Informe integrado de metodología y resultados; 19.188 Ampliación tabular y corrección de fórmulas del informe; 19.189 Auditoría exhaustiva y reconstrucción trazable del informe V2 |



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
