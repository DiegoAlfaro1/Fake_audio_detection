# Fake_audio_detection

Diego Alfaro - A01709971

Deteccion de audios generados con IA utilizando IA

Este proyecto fue creado y ejecutado utilizando kaggle

[Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

---

## Notebook `Fake_Audio_Detection_Model.ipynb`

Este notebook entrena un modelo de inteligencia artificial que "escucha" un audio de voz
y decide si es **real** (una persona de verdad) o **falso/fake** (generado por computadora).

### 1. Preparar las herramientas

Se cargan las librerías necesarias (TensorFlow para el modelo, librosa para el audio,
matplotlib para las gráficas) y se revisa si hay una tarjeta gráfica (GPU) disponible
para entrenar más rápido.

### 2. Convertir el sonido en una "imagen"

La computadora no entiende el sonido directamente, así que cada audio se transforma en un
**Mel Spectrogram**: una especie de imagen que muestra qué tonos suenan y en qué momento.

- Todos los audios se ajustan a **3 segundos**.
- Se les quita el silencio del inicio y el final.
- Se normaliza el volumen para que todos suenen parecido.

### 3. Organizar los datos

Se usan los audios del dataset **"The Fake or Real Dataset"** y se dividen en tres grupos:

- **Entrenamiento (training):** los audios con los que el modelo aprende.
- **Validación (validation):** sirven para revisar cómo va aprendiendo.
- **Prueba (testing):** audios diferentes (versión "rerec") que el modelo **nunca vio antes**,
  para comprobar que de verdad aprendió y no solo memorizó.

### 4. Crear variaciones de los audios

Para que el modelo sea más fuerte y no se confunda, durante el entrenamiento se le agregan
"dificultades" al sonido, por ejemplo:

- Ruido de fondo.
- Filtros que quitan algunas frecuencias.
- Eco (reverberación).
- Tapar pequeñas partes de la imagen del espectrograma.

Así el modelo aprende a reconocer la voz aunque el audio no esté perfecto.

### 5. El modelo

Es una combinación de dos tipos de red:

- **CNN:** buscan patrones en la imagen del audio.
- **BiLSTM:** analiza cómo cambia el sonido a lo largo del tiempo.

Al final, el modelo da un número entre 0 y 1:

- Cercano a **0 → audio falso (fake)**.
- Cercano a **1 → audio real**.

También se usan **pesos de clase** para arreglar el problema de que hay más audios de un tipo
que de otro, y así el modelo no se va "de lado".

### 6. Entrenamiento

El modelo aprende durante varias vueltas (épocas). Mientras entrena:

- Se guardan los mejores resultados automáticamente.
- Si deja de mejorar, se **detiene solo** para no perder tiempo.
- Si se estanca, baja la velocidad de aprendizaje para afinar mejor.

### 7. Evaluación

Al final se mide el desempeño con los audios de prueba usando varias métricas:

- **Accuracy:** qué porcentaje acertó.
- **Precision y Recall:** qué tan confiable es al decir "real" o "fake".
- **F1:** un balance entre las dos anteriores.
- **AUC y EER:** qué tan bien separa lo real de lo falso.
- **Matriz de confusión:** tabla que muestra los aciertos y los errores.

---

## Resultados

### Métricas del modelo

| Métrica    | Valor |
| ---------- | ----- |
| Accuracy   |       |
| Precision  |       |
| Recall     |       |
| F1 (macro) |       |
| AUC        |       |
| EER        |       |

### Gráficas

**Pérdida (Loss) y precisión (Accuracy) por época**

**Matriz de confusión**
