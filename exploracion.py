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
df.tail(20)

# procesamiento de la columna "categories"

print(df.categories.nunique())
print(df.categories.value_counts().to_string())

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
