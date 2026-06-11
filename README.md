# Sistema de recomendación de libros
Celina Medina Bucio, Alejandro Flores Martínez, Laura Thomas Godos

## Instrucciones para reproducción de experimentos
Debido a la naturaleza del algoritmo de Machine Learning que utilizamos, este siendo K-Nearest-Neighbors, no hay entrenamiento que realizar ni muchos experimentos que se hayan realizado. KNN no tiene una fase de entrenamiento en el sentido clásico, el modelo simplemente memoriza los datos y hace búsquedas en tiempo de inferencia.

- El archivo de **exploracion.py** incluye métodos de exploración inicial de la base de datos para comprender qué variables se tienen, de qué tipo son, y de qué manera están distribuidas
- El archivo de **preprocesamiento, pca y kmeans.py** cuenta con el preprocesamiento de los datos (por ejemplo, convertir las variables categóricas en columnas *dummy* para poder tratar con ellas numéricamente. Se incluye, también, el análisis de PCA y k-Means con sus visualizaciones relevantes.
- El archivo de **k_nn_y_visualizacion.py** es el análisis ya ahora sí de K-Nearest-Neighbors. Se incluye dentro de este documento también la interfaz generada para que el usuario interactúe con el sistema de recomendación.

Los tres archivos pueden correr de manera independiente de la otra, pero los tres requieren que el archivo de **data.csv** se encuentre en la misma carpeta que el propio archivo. **k_nn_y_visualizacion.py** requiere ser ejecutado en un Notebook de Google Colab, pues la librería de interfaz utilizada (ipywidget) es exclusiva para Jupyter Notebooks. 
