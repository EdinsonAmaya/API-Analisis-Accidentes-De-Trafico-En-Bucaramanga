# 🚦 API: Análisis y Predicción de Accidentes de Tránsito en Bucaramanga 📊

## 🌟 Resumen del Proyecto

Este proyecto se centra en el **análisis exploratorio y la predicción de la gravedad de accidentes de tránsito en la ciudad de Bucaramanga** (Colombia) utilizando datos históricos. El objetivo principal fue identificar patrones de incidencia por tiempo, lugar y tipo de vehículo, y aplicar modelos de Machine Learning para clasificar la severidad de los siniestros.

---

## 🔬 Metodología

El proceso de Ciencia de Datos aplicado se desarrolló en las siguientes etapas:

1.  **Carga y Limpieza de Datos:** Preparación inicial del conjunto de datos históricos de accidentes.
2.  **Análisis Exploratorio de Datos (EDA):** Identificación de patrones y tendencias de incidencia por factores clave (tiempo, lugar, tipo de vehículo).
3.  **Modelado:** Aplicación y evaluación de tres algoritmos de Machine Learning para la tarea de clasificación.

---

## 🧠 Modelos de Machine Learning Aplicados

Se aplicaron y evaluaron los siguientes tres algoritmos para predecir la gravedad de los accidentes, la cual fue dividida en clases como 'Con Muertos', 'Con Heridos' y 'Solo Daños'.

| Modelo | Descripción | Optimización |
| :--- | :--- | :--- |
| **K-Nearest Neighbors (KNN)** | Algoritmo de clasificación basado en la proximidad de puntos de datos. | Optimizado con **GridSearchCV** (parámetros `n_neighbors`, `weights`, `metric`). 
| **Árbol de Decisión (Decision Tree)** | Modelo de clasificación que construye un árbol de decisiones basado en características del dataset. | Optimizado con **GridSearchCV** (parámetros `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`). 
| **Naive Bayes (Gaussian Naive Bayes)** | Clasificador probabilístico basado en el Teorema de Bayes, asumiendo independencia entre características. | Entrenado con parámetros por defecto. 

---

## 📊 Resultados y Conclusiones

### Comparación de Accuracy

El modelo de **Árbol de Decisión** demostró ser el que tuvo el **mejor rendimiento general** en términos de *accuracy* , aunque todos los modelos mostraron un buen rendimiento general (alrededor del 82-83%).

| Modelo | Accuracy |
| :--- | :--- |
| **Árbol de Decisión** | **0.8314** | | **KNN** | **0.8175** |
| **Naive Bayes** | **0.6755** |

### Desafíos y Limitaciones

Una limitación significativa en todos los modelos fue la baja capacidad de predicción en las **clases minoritarias**, específicamente los accidentes con **víctimas mortales**.

*  El modelo **Naive Bayes** fue el **único** que logró identificar algunas de estas instancias raras ('Con Muertos'), aunque con muy baja confianza.
* Esto subraya la necesidad de aplicar **técnicas avanzadas para el manejo del desbalance de clases** (como SMOTE o ajuste de pesos) en futuras mejoras para predecir con fiabilidad estos eventos de alta gravedad pero baja frecuencia.

---

## 👩‍💻 Autores

Este proyecto fue desarrollado por estudiantes de la **Ingeniería de Sistemas** de la **Fundación Universitaria de San Gil (UNISANGIL)**.

* **Ederson Ferney Pico Santos** 
* **Camila Andrea Salazar Muñoz** 
* **Edinson Arley Amaya Ariza** 

**Facultad:** Ciencias Naturales e Ingeniería
**Programa:** Ingeniería de Sistemas / Énfasis en Ciencia de Datos 
**Lugar y Fecha:** San Gil, 2025-2 
