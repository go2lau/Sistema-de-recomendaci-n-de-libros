
# Exploración incial del dataset de Kaggle y selección de variables: Books Dataset que cuenta con información sobre libros

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("data.csv")

# Nuestra variable central a utilizar fue el promedio de los ratings, ya que es un indicador directo de lo mucho que la gente recomienda a un libro
# Title y authors se utilizan sólamente como identificadores y no se incluyen dentro del PCA. Categories sí se incluye, ya que saber qué categoría es la preferida por el usuario es crucial para poder hacerle recomendaciones adecuadas.


# Títulos más comunes


df["title"].value_counts().head(10)


# Es más probable que el modelo recomiende estos títulos (popularity bias).


# Autores con más libros


books=df.groupby("authors")["title"].count()


df.head()


# Filtrar por autor
autor = 'Stephen King'  # Cambia aquí
df_autor = df[df['authors'] == autor].copy()

# Agrupar por año — promedio de rating y número de libros
por_año = df_autor.groupby('published_year').agg(
    rating_promedio=('average_rating', 'mean'),
    num_libros=('title', 'count')
).reset_index()


fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(por_año['published_year'], por_año['rating_promedio'],
         marker='o', color='steelblue', linewidth=2, label='Rating promedio')
ax1.set_xlabel('Año de publicación')
ax1.set_ylabel('Rating promedio', color='steelblue')
ax1.set_ylim(1, 5)
ax1.tick_params(axis='y', labelcolor='steelblue')


ax2 = ax1.twinx()
ax2.bar(por_año['published_year'], por_año['num_libros'],
        alpha=0.3, color='orange', label='Libros publicados')
ax2.set_ylabel('Número de libros', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

plt.title(f'Rating de {autor} a lo largo del tiempo')
fig.tight_layout()
plt.show()




# Es más probable que un modelo recomiende a los autores con más libros. Debemos tomar esto en cuenta para prevenir posibles sesgos. Usamos como ejemplo a Stephen King para demostrar que el tiempo sí es una variable influyente en el rating y, por lo tanto, en lo recomendable que es un libro.


# Distribución del promedio de los ratings (con ponderación)



plt.hist(
    df['average_rating'],
    weights=df['ratings_count'],
)


# Distribución sin ponderación


plt.hist(
    df['average_rating'],
)


# La ponderación de los valores usando como pesos el número de ratings sí cambia la forma en la que estos se distribuyen, por lo cual se conservó al número de ratings como variable para análisis posteriores.


# Relación entre los ratings y el número de páginas


plt.scatter(df['average_rating'], df['num_pages'])


# Este scatterplot nos indica que la mayor cantidad de ratings entre 4 y 5 suelen tener menos de 500 páginas, lo cual es indicativo de que es mucho más probable que un libro se termine de leer y se califique si su número de páginas se encuentra dentro de este rango.




"""Libros con mejores ratings"""


ratings_books=df.groupby("title")["average_rating"].mean().sort_values(ascending=False)
ratings_books.head(10)
