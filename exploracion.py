# %% [markdown]
# Exploración incial del dataset de Kaggle y selección de variables: Books Dataset que cuenta con información sobre libros

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("data.csv")

df.title = np.where(df.subtitle.notnull(), df.title + '. ' + df.subtitle, df.title)
df = df.drop(columns = ["isbn13", 'isbn10', 'thumbnail', 'description', 'subtitle'])
df.ratings_count = df.ratings_count.fillna(df.ratings_count.mean())
df.published_year = df.published_year.fillna(df.published_year.median())
df.categories = df.categories.fillna("Unknown")
df.authors = df.authors.fillna("Unnamed")
df.title = df.title.fillna("Untitled")
df.tail(20)

# procesamiento de la columna "categories"

def cat_map(cat):
  cat = str(cat).lower().strip() # categorías sin espacios
  if any(word in cat for word in ["dystopias", "fiction", "novel", "tale", "chick lit", "stories", "humorous", "dystopias"] ):
    return "Fiction"
  if any(word in cat for word in ["juvenile", "young adult", "children", "boys", "girl", "babytime", "baby"]):
    return "Children, Juvenile and YA"
  if any(word in cat for word in["biographie", "biography", "autobiography", "authors"]):
    return "Biography and Autobiography"
  if any(word in cat for word in["history", "napoleon", "crusade", "revolution", "war", "stone age", "holocaust", "concentration camp", "presidents", "vice-presidents", "apartheid", "slave insurrections", "ancient"]):
    return "History"
  if any(word in cat for word in["star trek", "science fiction", "sci-fi", "interplanetary", "life on other planets", "mars", "human-alien encounters"]):
    return "Sci-Fi"
  if any(word in cat for word in ["sea monster", "fantasy", "dragon", "elf","elves", "magic", "imaginary place", "fictitious", "imaginary war"]):
    return "Fantasy"
  if any(word in cat for word in ["detective", "mystery", "espionage", "crime", "criminal", "murder", "assassin", "true crime"]):
    return "Mystery and Thriller"
  if any(word in cat for word in ["good and evil", "philosophy", "existential", "empiricism", "essentialism", "confucian", "fundamentalism"]):
    return "Philosophy"
  if any(word in cat for word in ["clergy", "jews", "monasteries", "religion", "theology", "god", "death", "spiritual life", "meditation", "christian",  "saints", "buddhis", "catholics", "bible", "church", "psychoanalysis and religion"]):
    return "Religion and Spirituality"
  if any(word in cat for word in ["mentally ill", "psychology", "alienation", "self-help", "identity", "courage", "identity", "repression", "autonomy", "adjustment"]):
    return "Psych and Self Help"
  if any(word in cat for word in ["advertising", "business", "consumer", "economy", "finance", "market", "enterprise", "businessmen"]):
    return "Business and Econ"
  if any(word in cat for word in ["human cloning", "aeronautics", "", "zoology", "agriculture", "mathematics", "number", "heat", "physicist", "astronomers", "physics", "cosmology", "arithmetic", ]):
    return "Science"
  if any(word in cat for word in ["computer", "technology", "emgineering", "programming", "c("]):
    return "Technology"
  if any(word in cat for word in ["museum", "ballet", "art", "performing arts", "artist", "actor", "photography", "architecture", "drawing", "costume"]):
    return "Art and Design"
  if any(word in cat for word in ["heroes", "antiheroes", "essay", "romance", "epic literature", "literature", "drama","literary criticism", "literary collections", "greek mythology", "poetry", "poet", "poem", "plays", "authors", "novelist"]):
    return "Literature"
  if any(word in cat for word in ["hitchhiking", "new england", "los angeles", "travel", "geography", "czech republic", "africa", "england", "london", "amazon", "france", "boston", "great britain", "paris", "cities", "town", "mississipi", "india", "egypt", "canada", "japan", "arctic regions", "europe", "australia", "latin america", "china", "botswana", "cambridge", "kyoto", "canterbury", "dominican republic", "cornwall", "austria", "new york", "israel", "united states", "illinois", "azerbaijan", "ireland", "portugal", "bosnia and herzegovina"]):
    return "Travel, countries, regions and continents"
  if any(word in cat for word in ["cookbook", "cooking", "recipe", "brewing", "candy", "chocolate"]):
    return "Food and recipes"
  if any(word in cat for word in ["childbirth", "black death", "amyotrophic lateral sclerosis", "handicap", "brain", "body", "health", "disab", "cancer", "obesity","fitness", "medical"]):
    return "Health"
  if any(word in cat for word in ["social science", "political science", "law", "capitalism", ]):
    return "Political and Social Sciences"
  if any(word in cat for word in ["married", "divorce", "teenager", "family life", "family", "families", "men", "relationships", "man-woman relationships", "women", "grandmother", "brother", "sister", "friendship", "adolescence", "birthparents", "parenthood"]):
    return "Family and Relationships"
  if any(word in cat for word in ["humor", "comedy", "wit"]):
    return "Comedy and Humor"
  if any(word in cat for word in ["vampire", "horror", "exorcism", "cult", "demon"]):
    return "Horror"
  if any(word in cat for word in ["pet", "bird", "dog", "labrador", "animal", "nature", "garden", "cat", "caterpillars"]):
    return "Nature, Plants and Animals"
  if any(word in cat for word in ["antique","music", "comic", "graphic novel", "games", "crafts", "hobby", "comic books", "sports", "recreation"] ):
    return "Entertainment and Graphic Novel"
  if any(word in cat for word in ["study aids", "education", "adult education", "language", "reference", "foreign language", "adult education"]):
    return "Education and Languages"
  if any(word in cat for word in ["unknown"]):
    return "Unknown"
  return "Other"

df["categories"] = df["categories"].apply(cat_map)
print(df["categories"].value_counts())
print("Sin categorizar:", (df["categories"] == "Other").sum())

# Corrected line: Filter the DataFrame for 'Fiction' category
df[df['categories'] == 'Fiction']

# %% [markdown]
# Nuestra variable central a utilizar fue el promedio de los ratings, ya que es un indicador directo de lo mucho que la gente recomienda a un libro.
# 
# Title y authors se utilizan sólamente como identificadores y no se incluyen dentro del PCA. Categories sí se incluye, ya que saber qué categoría es la preferida por el usuario es crucial para poder hacerle recomendaciones adecuadas.

# %% [markdown]
# Títulos más comunes

# %% [markdown]
# 

# %%
df["title"].value_counts().head(10)

# %% [markdown]
# Es más probable que el modelo recomiende estos títulos (popularity bias).

# %%


# %% [markdown]
# Autores con más libros

# %%
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



# %% [markdown]
# Es más probable que un modelo recomiende a los autores con más libros. Debemos tomar esto en cuenta para prevenir posibles sesgos. Usamos como ejemplo a Stephen King para demostrar que el tiempo sí es una variable influyente en el rating y, por lo tanto, en lo recomendable que es un libro.

# %% [markdown]
# Distribución del promedio de los ratings (con ponderación)
# 

# %%
plt.hist(
    df['average_rating'],
    weights=df['ratings_count'],
)

# %% [markdown]
# Distribución sin ponderación

# %%
plt.hist(
    df['average_rating'],
)

# %% [markdown]
# La ponderación de los valores usando como pesos el número de ratings sí cambia la forma en la que estos se distribuyen, por lo cual se conservó al número de ratings como variable para análisis posteriores.

# %% [markdown]
# Relación entre los ratings y el número de páginas

# %%
plt.scatter(df['average_rating'], df['num_pages'])

# %% [markdown]
# Este scatterplot nos indica que la mayor cantidad de ratings entre 4 y 5 suelen tener menos de 500 páginas, lo cual es indicativo de que es mucho más probable que un libro se termine de leer y se califique si su número de páginas se encuentra dentro de este rango.




"""Libros con mejores ratings"""


ratings_books=df.groupby("title")["average_rating"].mean().sort_values(ascending=False)
ratings_books.head(10)
