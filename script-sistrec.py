import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from surprise import KNNWithZScore, Dataset, SVD, Reader, NormalPredictor, BaselineOnly
from surprise.model_selection import train_test_split, cross_validate, KFold
from collections import defaultdict

# py -m pip install pandas

# Read csv file
df = pd.read_csv("ratings.csv")

# ------------------------------------------------------------------------------------------

# * Ejercicio 1.
# Realizar una exploración inicial indicando el número de valor vacíos (NA), el
# número de muestras duplicadas, y el número de usuarios, productos y
# puntuaciones que hay en el DataFrame creado (userId,movieId,rating,timestamp)
print("\n---------- Ejercicio 1 ----------\n")

print("Total number elements: " + str(np.count_nonzero(df)))
print("Number of empty values (NA): " + str(np.count_nonzero(df.isna().values)))
print("Number of duplicated samples: " + str(df.duplicated(keep=False).sum()))
print("Number of users: " + str(len(pd.unique(df["userId"]))))
print("Number of movies: " + str(len(pd.unique(df["movieId"]))))
print("Number of ratings: " + str(len(pd.unique(df["rating"]))))
print("Number of timestamps: " + str(len(pd.unique(df["timestamp"]))))

# ------------------------------------------------------------------------------------------

# * Ejercicio 2.
# Si eliminamos del dataset los productos con menos de 5 puntuaciones,
# ¿cuántos productos quedan? A continuación, sobre el dataset obtenido, si
# eliminamos los usuarios con menos de 10 puntuaciones, ¿cuántos
# usuarios quedan? ¿y cuántos productos? ¿de qué tamaño es ahora la matriz?
print("\n---------- Ejercicio 2 ----------\n")

print("Removing movies with less than 5 ratings...")
# df_movies_rating_less_5 = df.drop(df[df['rating'] < 5].index)
df = df.groupby("movieId").filter(lambda x: x["rating"].count() >= 5)
print("Number of movies: " + str(df.nunique()["movieId"]))

print("\nRemoving users with less than 10 ratings...")
df = df.groupby("userId").filter(lambda x: x["rating"].count() >= 10)
print("Number of users: " + str(len(pd.unique(df["userId"]))))

# Ordenando los campos por numero de valores ascendente podemos asegurarnos que no haya
# peliculas con menos de 5 puntuaciones o usuarios con menos de 10 puntuaciones
# print(df['movieId'].value_counts(ascending=True))
# print(df['userId'].value_counts(ascending=True))

print("\nSize of matrix: " + str(df.shape))

# ------------------------------------------------------------------------------------------

# * Ejercicio 3.
# Representar en un histograma el número de puntuaciones por usuario.
# Hacer lo mismo con el número de puntuaciones por producto.
print("\n---------- Ejercicio 3 ----------\n")

# Numero de puntuaciones por usuario
sns.histplot(df.groupby("userId", as_index=False)["rating"].count(), x="rating").set(
    xlabel="rating", ylabel="users"
)
plt.show()

# Numero de puntuaciones por pelicula
sns.histplot(df.groupby("movieId", as_index=False)["rating"].count(), x="rating").set(
    xlabel="rating", ylabel="movies"
)
plt.show()

# ------------------------------------------------------------------------------------------

# * Ejercicio 4
# Representar en un histograma la media de puntuaciones por usuario.
# Hacer lo mismo con la media de puntuaciones por producto.
print("\n---------- Ejercicio 4 ----------\n")

mean_df = df.groupby("userId", as_index=False)["rating"].mean()

# Media de puntuaciones por usuario
sns.histplot(df.groupby("userId", as_index=False)["rating"].mean(), x="rating").set(
    xlabel="rating", ylabel="users"
)
plt.show()

# Media de puntuaciones por pelicula
sns.histplot(df.groupby("movieId", as_index=False)["rating"].mean(), x="rating").set(
    xlabel="rating", ylabel="movies"
)
plt.show()

# ------------------------------------------------------------------------------------------

# * Ejercicio 5
# Representar en un diagrama de barras la distribución de las
# puntuaciones.
print("\n---------- Ejercicio 5 ----------\n")

sns.displot(df, x="rating")

plt.show()

# ------------------------------------------------------------------------------------------

# * Ejercicio 6
# Crear un objeto de tipo surprise.Dataset a partir del DataFrame empleado
# previamente. Considerando la tarea de predicción y las métricas RMSE y
# MAE, aplicar cross_validate (con k = 5). ¿Cuál es el algoritmo que mejor
# se comporta y por qué? Comparar el funcionamiento de los algoritmos.
# Justificar la respuesta.

# Algoritmos: SVD, NormalPredictor, BaselineOnly, KNNWithZScore.

# KNNWithZScore: k = 50, y min_k = 2. Se probarán las tres medidas de
# similitud vistas: cosine, msd y pearson.

# Los algoritmos se evaluarán usando cross-validation con k = 5, y el umbral de
# relevancia será de 4 (para la tarea de recomendación).
print("\n---------- Ejercicio 6 ----------\n")

reader = Reader(rating_scale=(1, 5))

dataset = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

# # SVD
# print("\n* SVD Algorithm\n")

# algo = SVD()
# # Run 5-fold cross-validation and print results
# cross_validate(algo, dataset, measures=["RMSE", "MAE"], cv=5, verbose=True)

# # NormalPredictor

# print("\n* NormalPredictor Algorithm\n")

# algo = NormalPredictor()
# # Run 5-fold cross-validation and print results
# cross_validate(algo, dataset, measures=["RMSE", "MAE"], cv=5, verbose=True)

# # BaselineOnly

# print("\n* BaselineOnly Algorithm\n")

# algo = BaselineOnly()
# # Run 5-fold cross-validation and print results
# cross_validate(algo, dataset, measures=["RMSE", "MAE"], cv=5, verbose=True)

# # KNNWithZScore msd

# print("\n* KNNWithZScore Algorithm msd\n")

# sim_options = {"name": "msd"}

# algo = KNNWithZScore(k=50, min_k=2, sim_options=sim_options)
# # Run 5-fold cross-validation and print results
# cross_validate(algo, dataset, measures=["RMSE", "MAE"], cv=5, verbose=True)

# # KNNWithZScore cosine

# print("\n* KNNWithZScore Algorithm cosine\n")

# sim_options = {"name": "cosine"}

# algo = KNNWithZScore(k=50, min_k=2, sim_options=sim_options)
# # Run 5-fold cross-validation and print results
# cross_validate(algo, dataset, measures=["RMSE", "MAE"], cv=5, verbose=True)

# # KNNWithZScore pearson

# print("\n* KNNWithZScore Algorithm pearson\n")

# sim_options = {"name": "pearson"}

# algo = KNNWithZScore(k=50, min_k=2, sim_options=sim_options)
# # Run 5-fold cross-validation and print results
# cross_validate(algo, dataset, measures=["RMSE", "MAE"], cv=5, verbose=True)

# ------------------------------------------------------------------------------------------

# * Paso 7.
# Con respecto a la tarea de recomendación, se tendrán en cuenta listas de
# recomendación de tamaños 1, 2, 5 y 10. ¿Cuál es el algoritmo que mejor
# se comporta en este caso? Justificar las respuestas tomando como base
# una gráfica de precision-recall. Para el cálculo de las métricas, se partirá
# del código disponible en:
# # https://github.com/NicolasHug/Surprise/blob/master/examples/precision_recall_at_k.py
print("\n---------- Ejercicio 7 ----------\n")


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:

        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def get_precision_recall_data(dataset_splitted, algo, size):
    # kf = KFold(n_splits=5)

    # print("* Precision-Recall SVD Algorithm list size \n")

    dfObj = pd.DataFrame(columns=["precision", "recall"])

    for trainset, testset in dataset_splitted:
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=size, threshold=4)

        # Precision and recall can then be averaged over all users
        precision_value = sum(prec for prec in precisions.values()) / len(precisions)
        print("Precision: " + str(precision_value))

        recall_value = sum(rec for rec in recalls.values()) / len(recalls)
        print("Recall: " + str(recall_value))

        dfObj = dfObj.append(
            {"precision": precision_value, "recall": recall_value}, ignore_index=True
        )
        print("\n")

    return dfObj


algorithms_list = [
    {
        "name": "SVD",
        "algo": SVD(),
        "col": (np.random.random(), np.random.random(), np.random.random()),
    },
    {
        "name": "NormalPredictor",
        "algo": NormalPredictor(),
        "col": (np.random.random(), np.random.random(), np.random.random()),
    },
    {
        "name": "BaselineOnly",
        "algo": BaselineOnly(),
        "col": (np.random.random(), np.random.random(), np.random.random()),
    },
    {
        "name": "KNNWithZScore msd",
        "algo": KNNWithZScore(k=50, min_k=2, sim_options={"name": "msd"}),
        "col": (np.random.random(), np.random.random(), np.random.random()),
    },
    {
        "name": "KNNWithZScore pearson",
        "algo": KNNWithZScore(k=50, min_k=2, sim_options={"name": "pearson"}),
        "col": (np.random.random(), np.random.random(), np.random.random()),
    },
    {
        "name": "KNNWithZScore cosine",
        "algo": KNNWithZScore(k=50, min_k=2, sim_options={"name": "cosine"}),
        "col": (np.random.random(), np.random.random(), np.random.random()),
    },
]

kf = KFold(n_splits=5)
for list_size in [1, 2, 5, 10]:
    print(dataset)
    data_splitted = kf.split(dataset)

    for algorithm in algorithms_list:
        print(
            "\nPrecision-Recall for algorithm"
            + str(algorithm["name"])
            + " for size "
            + str(list_size)
            + "\n"
        )
        dfObj = get_precision_recall_data(data_splitted, algorithm["algo"], list_size)
        dfObj = dfObj.sort_values("recall")

        print(dfObj.sort_values("recall"))
        # gca stands for 'get current axis'
        ax = plt.gca()
        dfObj.plot(
            x="recall",
            y="precision",
            # data=dfObj,
            color=algorithm["col"],
            label=str(algorithm["name"]),
            ax=ax,
        )

    plt.title("Size" + str(list_size))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()
