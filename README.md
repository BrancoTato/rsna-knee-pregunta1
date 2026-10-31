# Detección de anomalías de rodilla en RM con supervisión débil — Pregunta 1

Examen parcial · Redes Neuronales y Aprendizaje Profundo · UNI FIIS · 2026-II

Clasificación multietiqueta de **12 hallazgos** en resonancia magnética de rodilla usando el conjunto
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
Solo 58 de 4 407 estudios tienen etiquetas de un radiólogo; el resto se etiqueta a partir del
informe radiológico en texto libre (nueve idiomas) con un extractor propio.

## Resultados

| Entregable | Resultado |
|---|---|
| E1 · Extractor de etiquetas débiles | AUC 0,706 · exactitud CV 0,740 (baseline 0,672) · tasa de ruido 0,260 |
| E2 · CNN 2.5D desde cero | AUC macro 0,589 |
| E2 · ResNet-50 (ImageNet) | AUC macro **0,606** |
| E3 · F1 macro (umbral por prevalencia) | 0,406 (ResNet-50) · 0,428 (CNN) |
| E4 · Ablación | solo congelar el backbone supera el margen de error (−0,057) |
| E6 · Calibración | error de prevalencia 0,091 → 0,000 |
| E7 · Sesgo lingüístico | tasa de mención 0,554 (inglés) frente a 0,066 (griego) |

Validación sobre los 58 estudios gold; IC95 % del AUC macro ≈ ±0,047 (Hanley–McNeil).

## Estructura

```
.
├── notebooks/
│   └── rsna_knee_pregunta1.ipynb   notebook completo, ejecutado, con todas las salidas
├── informe/                        informe en PDF, formato NeurIPS
├── requirements.txt
└── README.md
```

| Sección del notebook | Contenido |
|---|---|
| 0 · 1 | Entorno, configuración y selección de dos planos por estudio |
| EDA | Idiomas, longitud, secciones, prevalencia, co-ocurrencia, heterogeneidad de imagen, auditoría y fuga |
| Entregable 1 | Extractor, validación contra los 58 gold, calibración, tasa de ruido |
| Entregable 2 | CNN 2.5D desde cero y ResNet-50 (32 canales: 16 sagitales + 16 coronales) |
| Entregable 3 | AUC-ROC macro y precisión/recall/F1 a umbral 0,5 y a umbral por prevalencia |
| Entregable 4 | Ablación: pérdida, muestreo, tamaño de datos y backbone congelado |
| Entregable 5 | Grad-CAM |
| Entregable 6 | Desbalance de clases y mitigación |
| Entregable 7 | Sesgo geográfico y lingüístico |

## Datos

Los datos **no se incluyen** en el repositorio (569,76 GB; su redistribución está limitada por las reglas de la competición).
Para acceder: crear una cuenta en Kaggle, aceptar las reglas de la competición y adjuntar el dataset al notebook.
El notebook los lee de `/kaggle/input/competitions/rsna-knee-abnormality-detection`.

## Ejecución (un único comando)

**Opción A — Kaggle (recomendada).** Importar `notebooks/rsna_knee_pregunta1.ipynb` (*File → Import Notebook*),
activar *Accelerator: GPU T4 x2* e *Internet: On*, y ejecutar:

```
Save Version → Save & Run All
```

Duración aproximada: 1,5 h. Todas las tablas, figuras y modelos quedan en `/kaggle/working`.

**Opción B — local**, con GPU CUDA y el dataset montado en la ruta indicada en `CFG['root']`:

```bash
pip install -r requirements.txt && jupyter nbconvert --to notebook --execute notebooks/rsna_knee_pregunta1.ipynb --output resultado.ipynb --ExecutePreprocessor.timeout=-1
```

## Reproducibilidad

Semilla 42 · GPU Tesla T4 · Python 3 · PyTorch 2.11 · transformers 5.16.1 · scikit-learn 1.6.1.
Los modelos de lenguaje (embeddings multilingües y NLLB-200) y los pesos de ResNet-50 se descargan en la primera ejecución.

## Autores

Danny Branco , Percy Marcovich , Daniel Eduardo , Henry Marcos , Cesar Elias
