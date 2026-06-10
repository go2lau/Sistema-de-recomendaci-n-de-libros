import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""Títulos más comunes"""

df["title"].value_counts().head(10)

"""Autores con más libros"""

books=df.groupby("authors")["title"].count()
books_sorted=books.sort_values(ascending=False)
books_sorted.head(10)

df.head()

"""Distribución del promedio de los ratings (con ponderación)

"""

plt.hist(
    df['average_rating'],
    weights=df['ratings_count'],
)

"""Distribución sin ponderación"""

plt.hist(
    df['average_rating'],
)

"""La ponderación de los valores usando como pesos el número de ratings sí cambia la forma en la que estos se distribuyen, por lo cual se conservó al número de ratings como variable para análisis posteriores."""


ratings=df.groupby("authors")["average_rating"].mean().sort_values(ascending=False)
ratings.head(10)

"""Relación entre los ratings y el número de páginas"""

plt.scatter(df['average_rating'], df['num_pages'])

"""Libros con mejores ratings"""


ratings_books=df.groupby("title")["average_rating"].mean().sort_values(ascending=False)
ratings_books.head(10)
