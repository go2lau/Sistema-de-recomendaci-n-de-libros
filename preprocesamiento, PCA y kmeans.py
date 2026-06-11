import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("data.csv")

# corregir valores NA y 
df.title = np.where(df.subtitle.notnull(), df.title + '. ' + df.subtitle, df.title)
df = df.drop(columns = ["isbn13", 'isbn10', 'thumbnail', 'description', 'subtitle'])
df.ratings_count = df.ratings_count.fillna(df.ratings_count.mean())
df.published_year = df.published_year.fillna(df.published_year.median())
df.categories = df.categories.fillna("Unknown")
df.authors = df.authors.fillna("Unnamed")
df.title = df.title.fillna("Untitled")
df.tail(20)

# procesamiento de la columna categories

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

# PRUEBA DE K NEAREST NEIGHBORS
# 1. Quedarse con 3 categorías
cats_deseadas = ['Fiction', 'History', 'Science'] 
df = df[df['categories'].isin(cats_deseadas)].reset_index(drop=True)

# 2. Extraer títulos antes de dropear columnas
titles = df['title'].values

# 3. Features
df['author_book_cnt'] = df['authors'].map(df['authors'].value_counts())
df = df.drop(columns = ['title', 'authors'])

features = ['published_year', 'average_rating', 'num_pages', 'ratings_count', 'author_book_cnt']
X = StandardScaler().fit_transform(df[features].values)

# 4. Labels
le = LabelEncoder()
Y = le.fit_transform(df['categories'])
book_labels = le.classes_

# 5. Muestra Estratificada
def stratified_sample(X, Y, n_per_class = 50, seed = 42):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(Y):
        pool = np.where(Y == c)[0]
        idx.extend(rng.choice(pool, size = min(n_per_class, len(pool)), replace = False))
    idx = np.array(idx)
    return X[idx], Y[idx]

def stratified_idx(Y, n_per_class=50, seed=42):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(Y):
        pool = np.where(Y == c)[0]
        idx.extend(rng.choice(pool, size = min(n_per_class, len(pool)), replace = False))
    return np.array(idx)

bx, by = stratified_sample(X, Y)
sample_titles = titles[stratified_idx(Y)] # muestreo estratificado para que las 3 muestras sean de clases diferentes

# 6. Selección de queries
rng = np.random.default_rng(42)
n_classes = len(book_labels)  # 3
query_amount = n_classes
neighbor_amount = 5

def select_queries(Y, n_classes):
    return np.array([rng.choice(np.where(Y == c)[0]) for c in range(n_classes)])

PALETTE = ['#E63946', '#F4A261', '#2A9D8F']

# 7. Plot de vecinos (distancia -> coseno)
query_idx = select_queries(by, n_classes)

fig, axes = plt.subplots(
    query_amount, neighbor_amount + 1, # se le añade 1 para no contarse a si mismo
    figsize=(16, query_amount * 2.5)
)
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('Books Dataset - 3 categorías y sus 5 vecinos más cercanos\n' 
    'Distancia Coseno', fontsize = 13, color = 'white', fontweight = 'bold', y = 1.01)

nn = NearestNeighbors(n_neighbors = neighbor_amount + 1, metric = 'cosine', n_jobs = -1)
nn.fit(bx)
distances, indices = nn.kneighbors(bx[query_idx])
distances = distances[:, 1:]
indices = indices[:, 1:]

for row in range(query_amount):
    q_class = int(by[query_idx[row]])
    color = PALETTE[q_class % len(PALETTE)]

    for col in range(neighbor_amount + 1):
        ax = axes[row][col]
        ax.set_facecolor('#16213e')
        ax.set_xticks([])
        ax.set_yticks([])

        lw = 2.5 if col == 0 else 0.8
        ec = color if col == 0 else '#555555' # color diferente para cada categoria
        for sp in ax.spines.values():
            sp.set_color(ec)
            sp.set_linewidth(lw)

        feat_names = ['year', 'rating', 'pages', 'n_ratings', 'author_cnt']
        if col == 0: # primer columna = valor aleatorio / muestra
            q_feats = bx[query_idx[row]]
            ax.barh(feat_names, q_feats, color=[color if v >= 0 else '#888888' for v in q_feats], height=0.6)
            ax.axvline(0, color = 'white', linewidth = 0.5, alpha = 0.4)
            ax.tick_params(axis = 'y', labelsize = 6, colors = 'black', pad = 1)
            short_title = (sample_titles[query_idx[row]][:22] + '…') if len(sample_titles[query_idx[row]]) > 22 else sample_titles[query_idx[row]]
            ax.set_title(f'CONSULTA\n{book_labels[q_class]}\n{short_title}', fontsize = 6, color = color, fontweight = 'bold', pad = 2)
        else:
            nb_idx  = indices[row, col - 1]
            nb_class = int(by[nb_idx])
            nb_dist = distances[row, col - 1]
            match_class = nb_class == q_class
            nb_color_bar = '#00FF88' if match_class else '#FF4444' # si el vecino concuerda con la categoria de la muesta

            ax.barh(feat_names, bx[nb_idx], color = [nb_color_bar if v >= 0 else '#888888' for v in bx[nb_idx]], height = 0.6, alpha = 0.85)
            ax.axvline(0, color = 'white', linewidth = 0.5, alpha = 0.4)
            ax.tick_params(axis = 'y', labelsize = 6, colors = 'white', pad = 1)

            short_nb = (sample_titles[nb_idx][:20] + '…') if len(sample_titles[nb_idx]) > 20 else sample_titles[nb_idx]
            ax.set_title(f'{book_labels[nb_class]}\nd={nb_dist:.3f}\n{short_nb}', fontsize=5.5, color=nb_color_bar, pad=2)

plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.show()
# categorías que quedaron dentro de other
print("============================================================")
df_raw = pd.read_csv("data.csv")
other_indices = df[df['categories'] == 'Other'].index
print(df_raw.loc[other_indices, 'categories'].value_counts().to_string())


# forma actual del df
print(df.dtypes)
print(df.shape)

# encode y arreglar las columnas que no son numéricas
df['author_book_cnt'] = df['authors'].map(df['authors'].value_counts())
df = df.drop(columns=['title', 'authors'])
df = pd.get_dummies(df, columns=['categories'])

df_scaled = StandardScaler().fit_transform(df)

pca = PCA(n_components=2)
pc = pca.fit_transform(df_scaled)
pcdf = pd.DataFrame(data = pc, columns = ['principal component 1', 'principal component 2'])
pcdf.head()

# elbow plot y silhouette para encontrar el número ideal de clústers
sse = {}
for k in range(2, 26):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_scaled)
    data_labels = kmeans.labels_
    sse[k] = kmeans.inertia_

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Número de clústers")
plt.ylabel("Error cuadrático medio")
plt.title("Elbow plot")
plt.show()

for cluster in range(2, 26):
  kmeans = KMeans(n_clusters=cluster).fit(df_scaled)
  data_labels = kmeans.labels_
  sil_coeff = silhouette_score(df_scaled, data_labels, random_state = 42, metric='euclidean')
  print("Silhouette coefficient for k = {}: {}".format(cluster, sil_coeff))
