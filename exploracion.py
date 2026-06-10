import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
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
