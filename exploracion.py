import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")
df.title = np.where(df.subtitle.notnull(), df.title + '. ' + df.subtitle, df.title)
df = df.drop(columns = ["isbn13", 'isbn10', 'thumbnail', 'description', 'subtitle'])
df.ratings_count = df.ratings_count.fillna(df.ratings_count.mean())
df.published_year = df.published_year.fillna(df.published_year.median())
df.categories = df.categories.fillna("Unknown")
df.authors = df.authors.fillna("Unnamed")
df.title = df.title.fillna("Untitled")
df.num_pages = df.num_pages.fillna(df.num_pages.median())
df.average_rating = df.average_rating.fillna(df.average_rating.median())
df.tail(20)


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
