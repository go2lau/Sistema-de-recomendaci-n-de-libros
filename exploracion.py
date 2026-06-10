import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
# =============================================================================
# 1. Cargar datos
# =============================================================================
 
df = pd.read_csv('data.csv')
 
# =============================================================================
# 2. Limpieza de datos
# =============================================================================
 
# Combinar title + subtitle y eliminar columnas innecesarias
df['title'] = np.where(df['subtitle'].notnull(), df['title'] + '. ' + df['subtitle'], df['title'])
df = df.drop(columns=['isbn13', 'isbn10', 'thumbnail', 'description', 'subtitle'])
 
# Rellenar valores faltantes
df['ratings_count'] = df['ratings_count'].fillna(df['ratings_count'].mean())
df['published_year'] = df['published_year'].fillna(df['published_year'].median())
df['categories'] = df['categories'].fillna('Unknown')
df['authors'] = df['authors'].fillna('Unnamed')
df['title'] = df['title'].fillna('Untitled')
 
print(df.tail(20))
 
# =============================================================================
# 3. Exploración de categorías originales
# =============================================================================
 
print('\nCategorías únicas:', df['categories'].nunique())
print(df['categories'].value_counts().to_string())
 
# =============================================================================
# 4. Mapeo de categorías
# =============================================================================
 
def cat_map(cat):
    cat = str(cat).lower().strip()
 
    if any(word in cat for word in ['fiction', 'novel', 'tale', 'narrative', 'chick lit', 'stories', 'humorous', 'dystopias']):
        return 'Fiction'
    if any(word in cat for word in ['juvenile', 'young adult', 'children', 'boys', 'girl', 'babytime', 'baby']):
        return 'Children, Juvenile and YA'
    if any(word in cat for word in ['biographie', 'biography', 'autobiography', 'authors']):
        return 'Biography and Autobiography'
    if any(word in cat for word in ['history', 'napoleon', 'crusade', 'revolution', 'war', 'holocaust', 'concentration camp', 'presidents', 'vice-presidents', 'apartheid', 'slave insurrections', 'ancient']):
        return 'History'
    if any(word in cat for word in ['star trek', 'science fiction', 'sci-fi', 'interplanetary', 'life on other planets', 'mars', 'human-alien encounters']):
        return 'Sci-Fi'
    if any(word in cat for word in ['fantasy', 'dragon', 'elf', 'magic', 'imaginary place', 'fictitious', 'imaginary war']):
        return 'Fantasy'
    if any(word in cat for word in ['detective', 'mystery', 'espionage', 'crime', 'criminal', 'murder', 'assassin', 'true crime']):
        return 'Mystery and Thriller'
    if any(word in cat for word in ['religion', 'god', 'death', 'spiritual life', 'meditation', 'christian saints', 'buddhis', 'catholics', 'bible', 'church', 'psychoanalysis and religion']):
        return 'Religion and Spirituality'
    if any(word in cat for word in ['psychology', 'self-help']):
        return 'Psych and Self Help'
    if any(word in cat for word in ['business', 'economy']):
        return 'Business and Econ'
    if any(word in cat for word in ['aeronautics', 'zoology', 'agriculture', 'mathematics', 'number', 'physics', 'cosmology']):
        return 'Science'
    if any(word in cat for word in ['computer', 'technology', 'emgineering', 'programming', 'c(']):
        return 'Technology'
    if any(word in cat for word in ['art', 'performing arts', 'photography', 'architecture', 'drawing', 'costume']):
        return 'Art and Design'
    if any(word in cat for word in ['literature', 'drama', 'literary criticism', 'literary collections', 'greek mythology', 'poetry', 'poem', 'plays']):
        return 'Literature'
    if any(word in cat for word in ['travel', 'geography', 'africa', 'england', 'london', 'boston', 'great britain', 'paris', 'india']):
        return 'Countries, regions and continents'
    if any(word in cat for word in ['cooking']):
        return 'Food and recipes'
    if any(word in cat for word in ['body', 'health', 'fitness', 'medical']):
        return 'Health'
    if any(word in cat for word in ['social science', 'political science', 'law']):
        return 'Political and Social Sciences'
    if any(word in cat for word in ['family', 'relationships']):
        return 'Family'
    if any(word in cat for word in ['humor', 'comedy']):
        return 'Comedy and Humor'
    if any(word in cat for word in ['pet']):
        return 'Nature, Plants and Animals'
    if any(word in cat for word in ['music', 'comic', 'graphic novel', 'games', 'crafts', 'hobby', 'comic books', 'sports', 'recreation']):
        return 'Entertainment and Graphic Novel'
    if any(word in cat for word in ['education', 'adult education', 'language', 'reference', 'foreign language']):
        return 'Education and Languages'
    if any(word in cat for word in ['unknown']):
        return 'Unknown'
    return 'Other'
 
df['categories'] = df['categories'].apply(cat_map)
 
print('\nCategorías después del mapeo:')
print(df['categories'].value_counts())
print('Sin categorizar:', (df['categories'] == 'Other').sum())
 
# =============================================================================
# 5. Títulos más frecuentes
# =============================================================================
 
print('\nTop 10 títulos más frecuentes:')
print(df['title'].value_counts().head(10))
 
# =============================================================================
# 6. Distribución de ratings (ponderada por número de reseñas)
# =============================================================================
 
plt.figure(figsize=(8, 5))
plt.hist(df['average_rating'], weights=df['ratings_count'], bins=20, color='steelblue', edgecolor='white')
plt.xlabel('Rating promedio')
plt.ylabel('Número de reseñas')
plt.title('Distribución de ratings (ponderada por reseñas)')
plt.tight_layout()
plt.show()
 
# =============================================================================
# 7. Autores con mejor rating promedio
# =============================================================================
 
ratings = df.groupby('authors')['average_rating'].mean().sort_values(ascending=False)
print('\nTop 10 autores por rating promedio:')
print(ratings.head(10))
