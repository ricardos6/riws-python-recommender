"""
This file is an example of how to evaluate using surprise

Evaluacion de algoritmo con surprise
"""

from surprise import accuracy, Dataset, SVD
from surprise.model_selection import train_test_split, cross_validate, KFold

# Load the movielens-100k dataset.
data = Dataset.load_builtin("ml-100k")

# Option 1

# print("****** Option 1 ******")

# Define a CF algorithm.
algo = SVD()

# trainset = data.build_full_trainset()

# algo.fit(trainset)

# uid = str(196)  # raw user id (as in the ratings file). They are strings!
# iid = str(845)  # raw item id (as in the ratings file). They are strings!

# # data.raw_ratings is a tuple (user, item, rating, timestamp) list
# rui = [r for (u, i, r, _) in data.raw_ratings if (u == uid) & (i == iid)]
# rui = rui[0] if len(rui) > 0 else None
    
# pred = algo.predict(uid, iid, r_ui = rui, verbose=True)

# Option 2
print ("\n****** Option 2 ******")

# sample random trainset and testset
# test set is made of 20% of the ratings.
trainset, testset = train_test_split(data, test_size=0.2)

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

# Option 3.1
print ("\n****** Option 3.1 ******")

# define a cross-validation iterator
kf = KFold(n_splits=5)

algo = SVD()

for i, (trainset_cv, testset_cv) in enumerate(kf.split(data)):

    print("Fold number", i + 1)
    
    # train and test algorithm.
    algo.fit(trainset_cv)
    
    print("On testset,", end="  ")
    
    predictions = algo.test(testset_cv)
    
    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
    
    # Se usa el conjunto de entrenamiento para la evaluacion
    print("On trainset,", end="  ")
    
    predictions = algo.test(trainset_cv.build_testset())
    # RMSE should be low as we are biased
    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)

# Option 3.2

print ("\n****** Option 3.2 ******")

algo = SVD()

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

# MAE: 