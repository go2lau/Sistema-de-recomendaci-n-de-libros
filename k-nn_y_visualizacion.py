import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from sklearn.neighbors import NearestNeighbors
df_raw = pd.read_csv("data.csv")
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(df_scaled)
titles = df_raw['title'].reset_index(drop = True)
authors = df_raw['authors'].reset_index(drop = True)

def rec_system(idx, k = 5): # recomendar 5 libros en base a índice de libro en base de datos
    distancias, indices = knn.kneighbors([df_scaled[idx]], n_neighbors = k+1)
    full_rec = titles.iloc[indices[0][1:]] + " - " + authors.iloc[indices[0][1:]]
    return full_rec

autocomplete = widgets.Text(
    placeholder = "Escribe el nombre de un libro...",
    layout = widgets.Layout(width='500px')
)
selection = widgets.BoundedIntText(
    description = 'Seleccionar #:',
    placeholder = 0,
    min = 1,
    layout=widgets.Layout(width='200px', display='none')  # oculto inicialmente
)
estado = {'coincidencias': None} # si sí hay coincidencias o no -> estado compartido entre input textual y numerico
output = widgets.Output()
def on_text_change(change):
    if change['type'] == 'change' and change['name'] == 'value': # básicamente: si cambia el valor
        with output:
            output.clear_output()
            texto = change['new'].strip()
            if len(texto) < 2:
                return
            coincidencias = titles[titles.str.contains(texto, case = False, na = False)]
            coincidencias = coincidencias.drop_duplicates()
            estado['coincidencias'] = coincidencias
            if len(coincidencias) == 0:
                print("No se encontraron títulos.")
                return
            if len(coincidencias) > 1: # si hay mas de un resultado -> mostrar opciones
                print(f"Títulos encontrados ({len(coincidencias)}):\n")
                for i, t in enumerate(coincidencias.head(10), 1):
                    autor = authors.iloc[coincidencias.index[i-1]] # <- para mostrar el autor en la búsqueda
                    print(f"  {i}. {t} - {autor}")
                selection.max = min(len(coincidencias), 10)
                selection.value = 1
                selection.layout.display = 'inline-flex'
                return

            # si hay exactamente uno -> recomendar directamente (sin el selector numerico)
            idx = coincidencias.index[0]
            recomendaciones = rec_system(idx)
            print(f"Recomendaciones para '{coincidencias.iloc[0]}':\n")
            for i, t in enumerate(recomendaciones, 1):
                print(f"  {i}. {t}")

def on_num_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        with output:
            output.clear_output()
            coincidencias = estado['coincidencias']
            if coincidencias is None:
                return
            seleccion = change['new'] - 1  # convertir entrada de usuario a arreglo (indice 1 -> indice 0)
            print(f"Títulos encontrados ({len(coincidencias)}):\n")
            for i, t in enumerate(coincidencias.head(10), 1):
                autor = authors.iloc[coincidencias.index[i-1]]
                print(f"  {i}. {t} - {autor}")
            idx = coincidencias.iloc[seleccion:seleccion+1].index[0]
            titulo = coincidencias.iloc[seleccion]
            autor_seleccionado = authors.iloc[idx]
            recomendaciones = rec_system(idx)
            print(f"\nRecomendaciones para '{titulo} - {autor_seleccionado}':\n")
            for i, t in enumerate(recomendaciones, 1):
                print(f"  {i}. {t}")

# observe() se llama automaticamente cada vez que cualquier propiedad del widget cambia
# las funciones on_text_change y on_num_change se llaman cuando esto pasa
autocomplete.observe(on_text_change)
selection.observe(on_num_change)

display(autocomplete, selection, output)
