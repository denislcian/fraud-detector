# 🛡️ Financial Fraud Detection AI (Guardian AI)

Este repositorio contiene un sistema de **Machine Learning Supervisado** diseñado para identificar transacciones fraudulentas en tiempo real. El proyecto aborda uno de los desafíos más críticos en fintech: el **desbalanceo extremo de datos**.



## 🎯 El Desafío Técnico
En el dataset analizado, el fraude representa únicamente el **0.58%** del volumen total de transacciones. Un modelo convencional fallaría en detectar estas anomalías. Para solucionar esto, implementamos:

1. **Ingeniería de Características (Feature Engineering):** Cálculo de distancias geográficas (Haversine) y perfiles temporales de gasto.
2. **SMOTE (Synthetic Minority Over-sampling):** Generación de datos sintéticos para equilibrar las clases y mejorar el aprendizaje del modelo.
3. **Optimización de Recall:** Configuración del modelo para priorizar la captura de fraudes, minimizando los falsos negativos.

## 🛠️ Stack Tecnológico
* **Core:** Python, Scikit-Learn.
* **Data Balancing:** Imbalanced-Learn (SMOTE).
* **Algoritmo:** Random Forest Classifier (Robusto frente a outliers y datos no lineales).
* **Deployment:** Streamlit para la interfaz de monitoreo en tiempo real.

## 🏗️ Arquitectura del Proyecto
```text
├── data/           # Datasets (Train/Test)
├── models/         # Modelos entrenados (.pkl)
├── notebooks/      # Investigación y validación de métricas
├── src/            # Motores de procesamiento y modelado
└── app.py          # Dashboard de detección