import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# py -m pip install pandas


# Read csv file
df = pd.read_csv('ratings.csv')
# * Paso 1.
# Realizar una exploración inicial indicando el número de valor vacíos (NA), el
# número de muestras duplicadas, y el número de usuarios, productos y
# puntuaciones que hay en el DataFrame creado (userId,movieId,rating,timestamp)

print("Total number elements: " + str(np.count_nonzero(df)))            
print("Number of empty values (NA): " + str(np.count_nonzero(df.isna().values)))
print("Number of duplicated samples: " + str(df.duplicated(keep=False).sum()))
print("Number of users: " + str(df.nunique()['userId']))
print("Number of movies: " + str(df.nunique()['movieId']))
print("Number of ratings: " + str(df.nunique()['rating']))
print("Number of timestamps: " + str(df.nunique()['timestamp']))

print("\n--------------------------------------------------------------------------\n")

# * Paso 2.
# Si eliminamos del dataset los productos con menos de 5 puntuaciones,
# ¿cuántos productos quedan? A continuación, sobre el dataset obtenido, si
# eliminamos los usuarios con menos de 10 puntuaciones, ¿cuántos
# usuarios quedan? ¿y cuántos productos? ¿de qué tamaño es ahora la matriz?

print("Removing movies with less than 5 ratings...")
# df_movies_rating_less_5 = df.drop(df[df['rating'] < 5].index)
df = df.groupby('movieId')
df = df.filter(lambda x: len(x) > 5)
print("Number of users: " + str(df.nunique()['userId']))
print("Number of movies: " + str(df.nunique()['movieId']))

print("\nRemoving users with less than 10 ratings...")
df = df.groupby('userId')
df = df.filter(lambda x: len(x) > 5)
print("Number of users: " + str(df.nunique()['userId']))
print("Number of movies: " + str(df.nunique()['movieId']))

# Ordenando los campos por numero de valores ascendente podemos asegurarnos que no haya
# peliculas con menos de 5 puntuaciones o usuarios con menos de 10 puntuaciones
# print(df['movieId'].value_counts(ascending=True))
# print(df['userId'].value_counts(ascending=True))

print("\nSize of matrix: " + str(df.shape))

print("\n--------------------------------------------------------------------------\n")
# * Paso 3.
# Representar en un histograma el número de puntuaciones por usuario.
# Hacer lo mismo con el número de puntuaciones por producto.

# Numero de puntuaciones por usuario
sns.displot(df, x="userId")

# Numero de puntuaciones por pelicula

sns.displot(df, x="movieId")

plt.show()

print("\n--------------------------------------------------------------------------\n")

# * Paso 4
# Representar en un histograma la media de puntuaciones por usuario.
# Hacer lo mismo con la media de puntuaciones por producto.

mean_df = df.groupby('userId', as_index=False)['rating'].mean()

# Miramos la tabla para ver que tiene sentido
# print(df.groupby('userId', as_index=False)['rating'].mean())

# Opcion a (seguramente este mal)
# sns.barplot(df.groupby('userId', as_index=False).mean('rating')[['userId', 'rating']], x='userId', y='rating')

# Opcion b
sns.displot(mean_df, x='rating').set(xlabel='rating', ylabel='count users')

sns.displot(mean_df, x='rating').set(xlabel='rating', ylabel='count movies')

plt.show()

print("\n--------------------------------------------------------------------------\n")

# * Paso 5
# Representar en un diagrama de barras la distribución de las
# puntuaciones.

# print(df)

sns.displot(df, x='rating')

plt.show()