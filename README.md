# Sistema de Monitoreo: Acciones Peligrosas en el Conductor - Microsueño

<div id="header" align="center">
  <img src="https://github.com/popcorner893/Sistema_Monitoreo_Conductor/blob/main/RecursosVisuales/Banner%20Inteligencia%20Artificial.png" />
</div>


## Autores

- [Iván Augusto Camargo López - 2230033](https://github.com/popcorner893)
- [Santiago Torres Barbosa - 2202024]

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
  <p>Ortega, J. D., Kose, N., Cañas, P., Chao, M.-A., Unnervik, A., Nieto, M., Otaegui, O., Salgado, L. (2020). DMD: A Large-Scale Multi-modal Driver Monitoring Dataset for Attention and Alertness Analysis [3].</p>
</div>


### Procesado - Dataset Propio 

Para este proyecto, se ha escogido la división **Drowsiness** del dataset, utilizando los videos desde la **cámara facial** para identificar signos de fatiga.

Con ayuda de Mediapipe **[4]** y ViTPose **[5]**, es posible crear una estructura de Pandas en donde las filas corresponden a *frames* de videos, y, las columnas, a *coordenadas de landmarks* detectados por estas herramientas. Juntando las coordenadas de los **468** landmarks faciales de Mediapipe (x,y,z), y **22** landmarks seleccionados de ViTPose (x,y), provenientes de las manos, además de 3 características a predecir: **blinks, eyes_state y yawning**, el subdataset de Drowsiness procesado cuenta con **92491 filas y 1453 columnas**, pertenecientes a 16 sujetos del DMD.

[Dataset Filtrado y Procesado](https://drive.google.com/file/d/181jpRp34J8gU2srIuj6HrFwkrssZrgEt/view?usp=sharing)

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

# Enlaces

- Código (Notebook final): [Link notebook final]()
- Video de explicación: [Link video]()
- Reposotorio: [Link repositorio](https://github.com/popcorner893/Sistema_Monitoreo_Conductor)

# Bibliografía

- Administración Nacional de Seguridad del Tráfico en las Carreteras (NHTSA). (2024). Manejar soñoliento.

- Ártimo. (2024, 21 de febrero). Microsueños y siniestros en Colombia: Una realidad preocupante.

- Boyacá Sie7e Días. (2023, 30 de junio). Fueron 8.032 los muertos en accidentes de tránsito el año pasado en Colombia. https://boyaca7dias.com.co/2023/06/30/fueron-8-032-los-muertos-en-accidentes-de-transito-el-ano-pasado-en-colombia

- Ortega, J., Kose, N., Cañas, P., Chao, M.a., Unnervik, A., Nieto, M., Otaegui, O., & Salgado, L. (2020). DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Alertness Analysis. In: A. Bartoli & A. Fusiello (eds), Computer Vision -- ECCV 2020 Workshops (pg. 387–405). Springer International Publishing.

- Lugaresi, C., Ma, S., Intwala, M., Manjunath, V., Lafleche, J. F., & Kossman, A. (2020). MediaPipe: A Framework for Perceiving and Processing Reality (arXiv:2006.10214). arXiv.

- Xu, Y., Du, Y., Zhang, W., Wang, Z., Wei, F., Lin, S., & Hu, R. (2022). ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation (arXiv:2204.12484). arXiv. https://arxiv.org/abs/2204.12484
