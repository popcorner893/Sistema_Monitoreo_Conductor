# Sistema de Monitoreo: Acciones Peligrosas en el Conductor - Microsueño

<div id="header" align="center">
  <img src="https://github.com/popcorner893/Sistema_Monitoreo_Conductor/blob/main/RecursosVisuales/Banner%20Inteligencia%20Artificial.png" />
</div>


## Autores

- [Iván Augusto Camargo López - 2230033](https://github.com/popcorner893)
- Santiago Torres Barbosa - 2202024

## Índice

- [Objetivo](#objetivo)
- [Motivación y Descripción del Problema](#motivación-y-descripción-del-problema)
- [Dataset](#dataset)
  - [Driver Monitoring Dataset - DMD](#driver-monitoring-dataset---dmd)
  - [Procesado - Dataset Propio](#procesado---dataset-propio)
- [Modelos](#modelos)
  - [Extracción de Posturas](#extracción-de-posturas)
  - [Machine Learning - Clasificación](#machine-learning---clasificación)
  - [Reducción de Dimensionalidad](#reducción-de-dimensionalidad)
  - [Clustering](#clustering)
- [Pipeline General](#pipeline-general)
- [Enlaces](#enlaces)
- [Bibliografía](#bibliografía)

## Objetivo

Integrar herramientas de Computer Vision (modelos de extracción de posturas como Mediapipe FaceMesh) y métodos tradicionales de Machine Learning para la detección de acciones peligrosas realizadas al volante, como microsueños, a partir de la clasificación de cuadros individuales provenientes de videos del rostro del conductor. Esto con el fin de alertar al conductor y pasajeros de vehículo, y evitar accidentes viales potencialmente fatales. 

## Motivación y Descripción del Problema

La fatiga y la distracción al volante son factores críticos en accidentes de tráfico a nivel mundial: conductas como la somnolencia afectan la concentración del conductor y aumentan el riesgo de colisión. 

En 2017, en los Estados Unidos se estimó que **91,000 accidentes** reportados por la policía involucraron a conductores somnolientos, resultando en aproximadamente 50,000 lesiones y casi 800 muertes. **[1]**

En **Colombia**, particularmente, los microsueños están vinculados a un alto porcentaje de siniestros viales **[2]**. En 2022 se registraron 8.032 muertes por accidentes de tránsito en Colombia, una de las cifras más altas de los últimos tiempos en materia de tránsito, lo que representó un aumento del 13,1% respecto al año anterior. Entre las causas mencionadas se encuentran la alta velocidad, imprudencias, cansancio, **microsueños**, fatiga y falta de experiencia **[3]**

Un sistema de monitoreo constante y en tiempo real permitiría al vehículo lanzar una señal de advertencia tras detectar indicios de somnolencia en el conductor, y, dado el caso, llevar a cabo medidas de seguridad; esto resulta especialmente importante en el marco de los nuevos automóviles inteligentes más recientes de la industria. este proyecto permite realizar una aproximación a la solución de esta problemática partiendo de un pipeline de procesamiento de datos y tareas de clasificación a través de algoritmos de ML.


## Dataset

### Driver Monitoring Dataset - DMD

El Driver Monitoring Dataset, de Vicomtech, destaca por ser uno de los datasets de monitoreo de conductor más variados y exhaustivos disponibles públicamente.

El dataset, en su conjunto, suma más de 25 TBs de datos, distribuidos en videos clasificados cuidadosamente de acuerdo a las acciones realizadas en cada video.

Se cuenta con videos de 37 participantes (27 hombres y 10 mujeres) y más de 20 etiquetas. Cada video cuenta con distintas perspectivas, canales y condiciones lumínicas de captura, dotando al dataset de una gran variedad para trabajar. 

Se divide al dataset internamente en 3 subsets: **Gaze (vista), Distraction (distracción) y Drowsiness (fatiga)**

[Dataset](https://dmd.vicomtech.org/#about)

<div align="center">
  <img src="https://github.com/popcorner893/Sistema_Monitoreo_Conductor/blob/main/RecursosVisuales/DMD_gif.gif" />
  <p>Ortega, J. D., Kose, N., Cañas, P., Chao, M.-A., Unnervik, A., Nieto, M., Otaegui, O., Salgado, L. (2020). DMD: A Large-Scale Multi-modal Driver Monitoring Dataset for Attention and Alertness Analysis [4].</p>
</div>


### Procesado - Dataset Propio 

Para este proyecto, se ha escogido la división **Drowsiness** del dataset, utilizando los videos desde la **cámara facial** para identificar signos de fatiga.

Con ayuda de Mediapipe **[4]** y ViTPose **[5]**, es posible crear una estructura de Pandas en donde las filas corresponden a *frames* de videos, y, las columnas, a *coordenadas de landmarks* detectados por estas herramientas. Juntando las coordenadas de los **468** landmarks faciales de Mediapipe (x,y,z), y **22** landmarks seleccionados de ViTPose (x,y), provenientes de las manos, además de 3 características a predecir: **blinks, eyes_state y yawning**, el subdataset de Drowsiness procesado cuenta con **92491 filas y 1453 columnas**, pertenecientes a 16 sujetos del DMD.

[Dataset Filtrado y Procesado](https://drive.google.com/file/d/19VynUTAJnvmAVz4Paar2gVssRljPX7Ni/view?usp=sharing)

## Modelos

### Extracción de Posturas

- Mediapipe FaceMesh
- ViTPose

### Machine Learning - Clasificación

- Gaussian Naive Bayes
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine
- Logistic Regression
- Kneighbors Classifier
- Linear Discriminant Analysis
- Deep Learning

### Reducción de Dimensionalidad

- PCA
- t-SNE
- Truncated SVD
- Factor Analysis

### Clustering

- K-Means
- DBSCAN
- Birch
- Gaussian Mixture

# Pipeline General

<div id="header" align="center">
  <img src="https://github.com/popcorner893/Sistema_Monitoreo_Conductor/blob/main/RecursosVisuales/Infografía%20Proyecto%20IA.png" />
</div>

# Resultados - Visualización

De manera local, se realizó el procesamiento de uno de los videos del DMD a partir de las predicciones de cada cuadro por separado (Véase Util_scripts). En este caso, se realizó una aplicación en la que se ponderan arbitrariamente las predicciones de las clases 'yawning', 'eyes_state' y 'blinks', y, a partir de sus ocurrencias en una ventana anterior de N frames, salta una alarma para simular la identificación de una conducta peligrosa en el conductor. 

<div align="center">
  <img src="https://github.com/popcorner893/Sistema_Monitoreo_Conductor/blob/main/RecursosVisuales/Demo_GIF.gif" />
  <p>Demo del Sistema. Utilización de video proveniente del DMD [4].</p>
</div>

# Limitaciones y Trabajo Futuro

- El sistema tiene un rendimiento aceptable cuando se evalúan las grabaciones procedentes del DMD, pero se está ligado a la posición relativa de la cámara con respecto al conductor; al grabar desde otro ángulo, las predicciones disminuyen su exactitud.
- Para solucionar esto, se podría considerar aplicar otras transformaciones y preprocesado del dataset, antes de pasar al aprendizaje. En el proyecto, se sugiere que transformar las coordenadas a otros espacios podría ser un buen punto de inicio.
- Se probó principalmente con la división Drowsiness del dataset DMD, dada la amplitud del dataset. Resultaría muy útil inspeccionar más a detalle las divisiones de Distriction y Gaze, para complementar la identificación de acciones peligrosas al volante.

# Conclusiones

- Las herramientas de extracción de posturas a través de videos (en este caso, Mediapipe FaceMesh y ViTPose), probaron ser un buen punto de partida para optar por un enfoque tabular en el análisis de videos que involucren ciertos comportamientos humanos (en este caso, aquellos relacionados al volante).
- Distintos modelos de ML tradicionales probaron ser efectivos a la hora de llevar a cabo la tarea de clasificación; destacando, entre ellos, la labor del RandomForestClassifier y el Kneighbors Classifier. Se puede atribuir el mejor rendimiento de estos últimos a la naturaleza de coordenadas de los datos y cómo estas se agrupan entre sí.
- Destaca la labor del aprendizaje no supervisado a través de los métodos de reducción de dimensionalidad. Específicamente, el PCA (Análisis de Componentes Principales) permitió mejorar el rendimiento y las métricas a través de la eliminación del ruido y el enfoque los componentes con mayor varianza.
- En cambio, los métodos de clustering implementados no fueron efectivos en la diferenciación de las clases. Un análisis visual a través de PCA o t-SNE permite observar que la separación de las etiquetas no es una labor trivial para nuestro caso de estudio. 

# Enlaces

- Código (Notebook final): [Link notebook final](https://drive.google.com/file/d/1gxYBAXHl9cVZc4irqw7EU25lF7-FZOff/view?usp=sharing)
- Video de explicación: [Link video]()
- Reposotorio: [Link repositorio](https://github.com/popcorner893/Sistema_Monitoreo_Conductor)

# Bibliografía

1. Administración Nacional de Seguridad del Tráfico en las Carreteras (NHTSA). (2024). Manejar soñoliento.

2. Ártimo. (2024, 21 de febrero). Microsueños y siniestros en Colombia: Una realidad preocupante.

3. Boyacá Sie7e Días. (2023, 30 de junio). Fueron 8.032 los muertos en accidentes de tránsito el año pasado en Colombia. https://boyaca7dias.com.co/2023/06/30/fueron-8-032-los-muertos-en-accidentes-de-transito-el-ano-pasado-en-colombia

4. Ortega, J., Kose, N., Cañas, P., Chao, M.a., Unnervik, A., Nieto, M., Otaegui, O., & Salgado, L. (2020). DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Alertness Analysis. In: A. Bartoli & A. Fusiello (eds), Computer Vision -- ECCV 2020 Workshops (pg. 387–405). Springer International Publishing.

5. Lugaresi, C., Ma, S., Intwala, M., Manjunath, V., Lafleche, J. F., & Kossman, A. (2020). MediaPipe: A Framework for Perceiving and Processing Reality (arXiv:2006.10214). arXiv.

6. Xu, Y., Du, Y., Zhang, W., Wang, Z., Wei, F., Lin, S., & Hu, R. (2022). ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation (arXiv:2204.12484). arXiv. https://arxiv.org/abs/2204.12484
