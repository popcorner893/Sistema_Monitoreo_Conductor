# Sistema de Monitoreo: Acciones Peligrosas en el Conductor - Microsueño

<div id="header" align="center">
  <img src="https://github.com/popcorner893/Sistema_Monitoreo_Conductor/blob/main/RecursosVisuales/Banner%20Inteligencia%20Artificial.png"
</div>

## Autores

- [Iván Augusto Camargo López - 2230033](https://github.com/popcorner893)
- [Santiago Torres Barbosa]

## Objetivo

Integrar herramientas de Computer Vision (modelos de extracción de posturas como Mediapipe FaceMesh) y métodos tradicionales de Machine Learning para la detección de acciones peligrosas realizadas al volante, como microsueños, a partir de la clasificación de cuadros individuales provenientes de videos del rostro del conductor. Esto con el fin de alertar al conductor y pasajeros de vehículo, y evitar accidentes viales potencialmente fatales. 

## Motivación y Descripción del Problema

La fatiga y la distracción al volante son factores críticos en accidentes de tráfico a nivel mundial: conductas como la somnolencia afectan la concentración del conductor y aumentan el riesgo de colisión. 

En 2017, en los Estados Unidos se estimó que **91,000 accidentes** reportados por la policía involucraron a conductores somnolientos, resultando en aproximadamente 50,000 lesiones y casi 800 muertes. [1]

En **Colombia**, particularmente, los microsueños están vinculados a un alto porcentaje de siniestros viales [2]. En 2022 se registraron 8.032 muertes por accidentes de tránsito en Colombia, una de las cifras más altas de los últimos tiempos en materia de tránsito, lo que representó un aumento del 13,1% respecto al año anterior. Entre las causas mencionadas se encuentran la alta velocidad, imprudencias, cansancio, **microsueños**, fatiga y falta de experiencia [3]

Un sistema de monitoreo constante y en tiempo real permitiría al vehículo lanzar una señal de advertencia tras detectar indicios de somnolencia en el conductor, y, dado el caso, llevar a cabo medidas de seguridad; esto resulta especialmente importante en el marco de los nuevos automóviles inteligentes más recientes de la industria. este proyecto permite realizar una aproximación a la solución de esta problemática partiendo de un pipeline de procesamiento de datos y tareas de clasificación a través de algoritmos de ML.


## Dataset

### Driver Monitoring Dataset - DMD

El Driver Monitoring Dataset, de Vicomtech, destaca por ser uno de los datasets de monitoreo de conductor más variados y exhaustivos disponibles públicamente.

El dataset, en su conjunto, suma más de 25 TBs de datos, distribuidos en videos clasificados cuidadosamente de acuerdo a las acciones realizadas en cada video.

Se cuenta con videos de 37 participantes (27 hombres y 10 mujeres) y más de 20 etiquetas. Cada video cuenta con distintas perspectivas, canales y condiciones lumínicas de captura, dotando al dataset de una gran variedad para trabajar. 

Se divide al dataset internamente en 3 subsets: **Gaze (vista), Distraction (distracción) y Drowsiness (fatiga)**

[Dataset](https://dmd.vicomtech.org/#about)

### Procesado - Dataset Propio 

## Pipeline General

<div id="header" align="center">
  <img src="https://github.com/popcorner893/Sistema_Monitoreo_Conductor/blob/main/RecursosVisuales/Infografía%20Proyecto%20IA.png"
</div>

# Bibliografía

- Administración Nacional de Seguridad del Tráfico en las Carreteras (NHTSA). (2024). Manejar soñoliento.

- Ártimo. (2024, 21 de febrero). Microsueños y siniestros en Colombia: Una realidad preocupante.

- Boyacá Sie7e Días. (2023, 30 de junio). Fueron 8.032 los muertos en accidentes de tránsito el año pasado en Colombia. https://boyaca7dias.com.co/2023/06/30/fueron-8-032-los-muertos-en-accidentes-de-transito-el-ano-pasado-en-colombia

- Ortega, J., Kose, N., Cañas, P., Chao, M.a., Unnervik, A., Nieto, M., Otaegui, O., & Salgado, L. (2020). DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Alertness Analysis. In: A. Bartoli & A. Fusiello (eds), Computer Vision -- ECCV 2020 Workshops (pg. 387–405). Springer International Publishing.

- Lugaresi, C., Ma, S., Intwala, M., Manjunath, V., Lafleche, J. F., & Kossman, A. (2020). MediaPipe: A Framework for Perceiving and Processing Reality (arXiv:2006.10214). arXiv.

- Xu, Y., Du, Y., Zhang, W., Wang, Z., Wei, F., Lin, S., & Hu, R. (2022). ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation (arXiv:2204.12484). arXiv. https://arxiv.org/abs/2204.12484

# Dataset
DMD - Driving Monitoring Dataset - Alrededor de 41 horas de grabaciones segmentadas en categorías, altamente variadas. 
https://dmd.vicomtech.org/#about

![alt text](https://www.kienyke.com/sites/default/files/2023-06/Microsuen%CC%83o.jpg)
