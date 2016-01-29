
# coding: utf-8

# #### *Part 0*: Preliminaries
# #### *Part 1*: Basic Recommendations
# #### *Part 2*: Collaborative Filtering
# #### *Part 3*: Predictions for Yourself
# [mllib]: https://spark.apache.org/mllib/

# ### Code

from pyspark import SparkContext
import sys
import time
import os
#from test_helper import Test

#baseDir = os.path.join('data')
#inputPath = os.path.join('cs100', 'lab4', 'small')

#ratingsFilename = os.path.join(baseDir, inputPath, 'ratings.dat.gz')
#moviesFilename = os.path.join(baseDir, inputPath, 'movies.dat')


# ### **Part 0: Preliminaries**
# #### We read in each of the files and create an RDD consisting of parsed lines.
# #### Each line in the ratings dataset (`ratings.dat.gz`) is formatted as:
# ####   `UserID::MovieID::Rating::Timestamp`
# #### Each line in the movies (`movies.dat`) dataset is formatted as:
# ####   `MovieID::Title::Genres`
# #### The `Genres` field has the format
# ####   `Genres1|Genres2|Genres3|...`
# #### The format of these files is uniform and simple, so we can use Python [`split()`](https://docs.python.org/2/library/stdtypes.html#str.split) to parse their lines.
# #### Parsing the two files yields two RDDS
# * #### For each line in the ratings dataset, we create a tuple of (UserID, MovieID, Rating). We drop the timestamp because we do not need it.
# * #### For each line in the movies dataset, we create a tuple of (MovieID, Title). We drop the Genres because we do not need them.


#numPartitions = 2
sc=SparkContext(appName="SparkPi")
rawRatings = sc.textFile('/gpfs/courses/cse603/students/dnazaret/722/ml-10M100K/ratings.dat')#.repartition(numPartitions)
rawMovies = sc.textFile('/gpfs/courses/cse603/students/dnazaret/722/ml-10M100K/movies.dat')

def get_ratings_tuple(entry):
    """ Parse a line in the ratings dataset
    Args:
        entry (str): a line in the ratings dataset in the form of UserID::MovieID::Rating::Timestamp
    Returns:
        tuple: (UserID, MovieID, Rating)
    """
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2])


def get_movie_tuple(entry):
    """ Parse a line in the movies dataset
    Args:
        entry (str): a line in the movies dataset in the form of MovieID::Title::Genres
    Returns:
        tuple: (MovieID, Title)
    """
    items = entry.split('::')
    return int(items[0]), items[1]

#As these RDD's would be used often we will cache them
ratingsRDD = rawRatings.map(get_ratings_tuple).cache()
moviesRDD = rawMovies.map(get_movie_tuple).cache()

def sortFunction(tuple):
    """ Construct the sort string (does not perform actual sorting)
    Args:
        tuple: (rating, MovieName)
    Returns:
        sortString: the value to sort with, 'rating MovieName'
    """
    key = unicode('%.3f' % tuple[0])
    value = tuple[1]
    return (key + ' ' + value)



# ### **Part 1: Basic Recommendations**
# #### One way to recommend movies is to always recommend the movies with the highest average rating. In this part, we will use Spark to find the name, number of ratings, and the average rating of the 20 movies with the highest average rating and more than 500 reviews. We want to filter our movies with high ratings but fewer than or equal to 500 reviews because movies with few reviews may not have broad appeal to everyone.


def getCountsAndAverages(IDandRatingsTuple):
    """ Calculate average rating
    Args:
        IDandRatingsTuple: a single tuple of (MovieID, (Rating1, Rating2, Rating3, ...))
    Returns:
        tuple: a tuple of (MovieID, (number of ratings, averageRating))
    """
    count = len(IDandRatingsTuple[1])
    avg = float(sum(IDandRatingsTuple[1]))/count
    countAvgTup = (count,avg)
    return (IDandRatingsTuple[0],countAvgTup)



# #### **(1b) Movies with Highest Average Ratings**
# #### Now that we have a way to calculate the average ratings, we will use the `getCountsAndAverages()` helper function with Spark to determine movies with highest average ratings.

# From ratingsRDD with tuples of (UserID, MovieID, Rating) we create an RDD with tuples of (MovieID, iterable of Ratings for that MovieID)
movieIDsWithRatingsRDD = (ratingsRDD
                          .map(lambda s:(s[1],s[2]))).groupByKey().map(lambda s: (s[0],list(s[1])))

# Using `movieIDsWithRatingsRDD`, we compute the number of ratings and average rating for each movie to yield 
#tuples of the form (MovieID, (number of ratings, average rating))
movieIDsWithAvgRatingsRDD = movieIDsWithRatingsRDD.map(getCountsAndAverages)
# To `movieIDsWithAvgRatingsRDD`, we apply RDD transformations that use `moviesRDD` to get the movie
# names for `movieIDsWithAvgRatingsRDD`, yielding tuples of the form
# (average rating, movie name, number of ratings)
movieNameWithAvgRatingsRDD = (moviesRDD
                              .join(movieIDsWithAvgRatingsRDD)
                              .map(lambda s:(s[1][1][1],s[1][0],s[1][1][0])))
bestmovie = (moviesRDD
                              .join(movieIDsWithAvgRatingsRDD)
                              .map(lambda s:(s[1][1][1],s[1][0],s[1][1][0],s[0])).filter(lambda s:s[2] > 500 )
                              .sortBy(sortFunction, False)).map(lambda s:(s[0],s[1],s[3]))


# We Apply an RDD transformation to `movieNameWithAvgRatingsRDD` to limit the results to movies with
# ratings from more than 500 people. We then use the `sortFunction()` helper function to sort by the
# average rating to get the movies in order of their rating (highest rating first)
movieLimitedAndSortedByRatingRDD = (movieNameWithAvgRatingsRDD
                                    .filter(lambda s:s[2] > 500 )
                                    .sortBy(sortFunction, False))
print 'Twenty Movies with highest ratings: %s' % movieLimitedAndSortedByRatingRDD.take(20)



# #### Using a threshold on the number of reviews is one way to improve the recommendations, but there are many other good ways to improve quality. For example, you could weight ratings by the number of ratings.

# ## **Part 2: Collaborative Filtering**
# #### We are going to use a technique called [collaborative filtering][collab]. Collaborative filtering is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). The underlying assumption of the collaborative filtering approach is that if a person A has the same opinion as a person B on an issue, A is more likely to have B's opinion on a different issue x than to have the opinion on x of a person chosen randomly.
# #### The image below (from [Wikipedia][collab]) shows an example of predicting of the user's rating using collaborative filtering. At first, people rate different items (like videos, images, games). After that, the system is making predictions about a user's rating for an item, which the user has not rated yet. These predictions are built upon the existing ratings of other users, who have similar ratings with the active user. For instance, in the image below the system has made a prediction, that the active user will not like the video.
# ![collaborative filtering](https://courses.edx.org/c4x/BerkeleyX/CS100.1x/asset/Collaborative_filtering.gif)
# [mllib]: https://spark.apache.org/mllib/
# [collab]: https://en.wikipedia.org/?title=Collaborative_filtering
# [collab2]: http://recommender-systems.org/collaborative-filtering/

# #### For movie recommendations, we start with a matrix whose entries are movie ratings by users.  Each column represents a user and each row represents a particular movie (shown in blue).
# #### Since not all users have rated all movies, we do not know all of the entries in this matrix, which is precisely why we need collaborative filtering.  For each user, we have ratings for only a subset of the movies.  With collaborative filtering, the idea is to approximate the ratings matrix by factorizing it as the product of two matrices: one that describes properties of each user (shown in green), and one that describes properties of each movie (shown in blue).
# ![factorization](http://spark-mooc.github.io/web-assets/images/matrix_factorization.png)
# #### We want to select these two matrices such that the error for the users/movie pairs where we know the correct ratings is minimized.  The [Alternating Least Squares][als] algorithm does this by first randomly filling the users matrix with values and then optimizing the value of the movies such that the error is minimized.  Then, it holds the movies matrix constrant and optimizes the value of the user's matrix.  This alternation between which matrix to optimize is the reason for the "alternating" in the name.
# #### Given a fixed set of user factors (i.e., values in the users matrix), we use the known ratings to find the best values for the movie factors using the optimization written at the bottom of the figure.  Then we "alternate" and pick the best user factors given fixed movie factors.
# [als]: https://en.wikiversity.org/wiki/Least-Squares_Method

# #### **(2a) Creating a Training Set**
# #### Before we jump into using machine learning, we need to break up the `ratingsRDD` dataset into three pieces:
# * #### A training set (RDD), which we will use to train models
# * #### A validation set (RDD), which we will use to choose the best model
# * #### A test set (RDD), which we will use for our experiments
# #### To randomly split the dataset into the multiple groups, we can use the pySpark [randomSplit()](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.randomSplit) transformation. `randomSplit()` takes a set of splits and and seed and returns multiple RDDs.

# In[12]:

trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L)


# #### **(2b) Root Mean Square Error (RMSE)**
# #### In the next part, we will generate a few different models, and will need a way to decide which model is best. We will use the [Root Mean Square Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE) or Root Mean Square Deviation (RMSD) to compute the error of each model.  RMSE is a frequently used measure of the differences between values (sample and population values) predicted by a model or an estimator and the values actually observed. The RMSD represents the sample standard deviation of the differences between predicted values and observed values. These individual differences are called residuals when the calculations are performed over the data sample that was used for estimation, and are called prediction errors when computed out-of-sample. The RMSE serves to aggregate the magnitudes of the errors in predictions for various times into a single measure of predictive power. RMSE is a good measure of accuracy, but only to compare forecasting errors of different models for a particular variable and not between variables, as it is scale-dependent.
# ####  The RMSE is the square root of the average value of the square of `(actual rating - predicted rating)` for all users and movies for which we have the actual rating. Versions of Spark MLlib beginning with Spark 1.4 include a [RegressionMetrics](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RegressionMetrics) modiule that can be used to compute the RMSE. However, since we are using Spark 1.3.1, we will write our own function.

import math

def computeError(predictedRDD, actualRDD):
    """ Compute the root mean squared error between predicted and actual
    Args:
        predictedRDD: predicted ratings for each movie and each user where each entry is in the form
                      (UserID, MovieID, Rating)
        actualRDD: actual ratings where each entry is in the form (UserID, MovieID, Rating)
    Returns:
        RSME (float): computed RSME value
    """
    # Transform predictedRDD into the tuples of the form ((UserID, MovieID), Rating)
    predictedReformattedRDD = predictedRDD.map(lambda s:((s[0],s[1]),s[2]))

    # Transform actualRDD into the tuples of the form ((UserID, MovieID), Rating)
    actualReformattedRDD = actualRDD.map(lambda s:((s[0],s[1]),s[2]))

    # Compute the squared error for each matching entry (i.e., the same (User ID, Movie ID) in each
    # RDD) in the reformatted RDDs using RDD transformtions
    squaredErrorsRDD = (predictedReformattedRDD
                        .join(actualReformattedRDD).map(lambda s:(math.pow((s[1][1]-s[1][0]),2))))
    
    # Compute the total squared error
    totalError = squaredErrorsRDD.reduce(lambda a,b:a+b)

    # Count the number of entries for which we computed the total squared error
    numRatings = squaredErrorsRDD.count()

    # Using the total squared error and the number of entries, compute the RSME
    return math.pow(totalError/numRatings,0.5)



# #### In this part, we will use the MLlib implementation of Alternating Least Squares, [ALS.train()](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS). ALS takes a training dataset (RDD) and several parameters that control the model creation process. To determine the best values for the parameters, we will use ALS to train several models, and then we will select the best model and use the parameters from that model.
# #### The process we will use for determining the best model is as follows:
# * #### Pick a set of model parameters. The most important parameter to `ALS.train()` is the *rank*, which is the number of rows in the Users matrix (green in the diagram above) or the number of columns in the Movies matrix (blue in the diagram above). (In general, a lower rank will mean higher error on the training dataset, but a high rank may lead to [overfitting](https://en.wikipedia.org/wiki/Overfitting).)  We will train models with ranks of 4, 8, and 12 using the `trainingRDD` dataset.
# * #### Create a model using `ALS.train(trainingRDD, rank, seed=seed, iterations=iterations, lambda_=regularizationParameter)` with three parameters: an RDD consisting of tuples of the form (UserID, MovieID, rating) used to train the model, an integer rank (4, 8, or 12), a number of iterations to execute (we will use 5 for the `iterations` parameter), and a regularization coefficient (we will use 0.1 for the `regularizationParameter`).
# * #### For the prediction step, create an input RDD, `validationForPredictRDD`, consisting of (UserID, MovieID) pairs that you extract from `validationRDD`. You will end up with an RDD of the form: `[(1, 1287), (1, 594), (1, 1270)]`
# * #### Using the model and `validationForPredictRDD`, we can predict rating values by calling [model.predictAll()](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.MatrixFactorizationModel.predictAll) with the `validationForPredictRDD` dataset, where `model` is the model we generated with ALS.train().  `predictAll` accepts an RDD with each entry in the format (userID, movieID) and outputs an RDD with each entry in the format (userID, movieID, rating).
# * #### We will evaluate the quality of the model by using the `computeError()` function we wrote in part (2b) to compute the error between the predicted ratings and the actual ratings in `validationRDD`.

from pyspark.mllib.recommendation import ALS
start=time.time()
validationForPredictRDD = validationRDD.map(lambda s:(s[0],s[1]))

seed = 5L
iterations = 5
regularizationParameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

minError = float('inf')
bestRank = -1
bestIteration = -1
for rank in ranks:
    model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
    predictedRatingsRDD = model.predictAll(validationForPredictRDD)
    error = computeError(predictedRatingsRDD, validationRDD)
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < minError:
        minError = error
        bestRank = rank

print 'The best model was trained with rank %s' % bestRank


# #### **(2d) Testing the Model**

# * #### Train a model, using the `trainingRDD`, `bestRank` from part (2c), and the parameters you used in in part (2c): `seed=seed`, `iterations=iterations`, and `lambda_=regularizationParameter`.
# * #### For the prediction step, create an input RDD, `testForPredictingRDD`, consisting of (UserID, MovieID) pairs that you extract from `testRDD`. You will end up with an RDD of the form: `[(1, 1287), (1, 594), (1, 1270)]`
# * #### Use [myModel.predictAll()](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.MatrixFactorizationModel.predictAll) to predict rating values for the test dataset.

myModel = ALS.train(trainingRDD, bestRank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
testForPredictingRDD = testRDD.map(lambda s:(s[0],s[1]))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)

testRMSE = computeError(testRDD, predictedTestRDD)
end=time.time()
print 'The model had a RMSE on the test set of %s' % testRMSE
print 'Total time for ALS training and RMSE computation was '+str(end-start)


# #### **(2e) Comparing the Model**
# #### Looking at the RMSE for the results predicted by the model versus the values in the test set is one way to evalute the quality of our model. Another way to evaluate the model is to evaluate the error from a test set where every rating is the average rating for the training set.

trainingAvgRating = (trainingRDD.map(lambda s:s[2]).reduce(lambda a,b:a+b))/trainingRDD.count()
print 'The average rating for movies in the training set is %s' % trainingAvgRating

testForAvgRDD = testRDD.map(lambda s:(s[0],s[1],trainingAvgRating))
testAvgRMSE = computeError(testRDD, testForAvgRDD)
print 'The RMSE on the average set is %s' % testAvgRMSE



# ## **Part 3: Predictions for new users**


print 'Most rated movies:'
print '(average rating, movie name, number of reviews)'
for ratingsTuple in bestmovie.take(50):
    print ratingsTuple


# #### The user ID 0 is unassigned, so we will use it for your ratings. We set the variable `myUserID` to 0 for you. Next, create a new RDD `myRatingsRDD` with your ratings for at least 10 movie ratings. Each entry should be formatted as `(myUserID, movieID, rating)` (i.e., each entry should be formatted in the same way as `trainingRDD`).  As in the original dataset, ratings should be between 1 and 5 (inclusive). If you have not seen at least 10 of these movies, you can increase the parameter passed to `take()` in the above cell until there are 10 movies that you have seen (or you can also guess what your rating would be for movies you have not seen).

myUserID = 0

# Note that the movie IDs are the *last* number on each line. A common error was to use the number of ratings as the movie ID.
myRatedMovies = [
     (0,318,1),(0,1,1),(0,296,1),(0,527,1),(0,2571,1),(0,2762,1),(0,1036,1),(0,589,1),(0,3578,1),(0,480,1)
     # The format of each line is (myUserID, movie ID, your rating)
     # For example, to give the movie "Star Wars: Episode IV - A New Hope (1977)" a five rating, we would add the following line:
     #   (myUserID, 260, 5),
    ]
myRatingsRDD = sc.parallelize(myRatedMovies)

# #### **(3b) Adding new user Movies to Training Dataset**
# #### Now that you have ratings for yourself, we need to add your ratings to the `training` dataset so that the model you train will incorporate your preferences.

trainingWithMyRatingsRDD = trainingRDD.union(myRatingsRDD)

# #### **(3c) Training a Model with these Ratings**

# TODO: Replace <FILL IN> with appropriate code
myRatingsModel = ALS.train(trainingWithMyRatingsRDD, bestRank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)

# #### **(3d) Checking RMSE for the New Model with our Ratings**

predictedTestMyRatingsRDD = myRatingsModel.predictAll(testForPredictingRDD)
testRMSEMyRatings = computeError(predictedTestMyRatingsRDD,testRDD)
print 'The model had a RMSE on the test set of %s' % testRMSEMyRatings


# #### **(3e) Predict Your Ratings**
# #### So far, we have only used the `predictAll` method to compute the error of the model.  Here, use the `predictAll` to predict what ratings we would give to the movies that you did not already provide ratings for.

# Use the Python list myRatedMovies to transform the moviesRDD into an RDD with entries that are pairs of the form (myUserID, Movie ID) and that does not contain any movies that you have rated.
moviesSeen = myRatingsRDD.map(lambda s:s[1]).collect()
myUnratedMoviesRDD = (moviesRDD
                      .map(lambda s:(0,s[0])).filter(lambda s:s[1] not in moviesSeen))

# Use the input RDD, myUnratedMoviesRDD, with myRatingsModel.predictAll() to predict your ratings for the movies
predictedRatingsRDD = myRatingsModel.predictAll(myUnratedMoviesRDD)
movieCountsRDD = movieIDsWithAvgRatingsRDD.map(lambda s:(s[0],s[1][0]))

# Transform predictedRatingsRDD into an RDD with entries that are pairs of the form (Movie ID, Predicted Rating)
predictedRDD = predictedRatingsRDD.map(lambda s:(s[1],s[2]))

# Use RDD transformations with predictedRDD and movieCountsRDD to yield an RDD with tuples of the form (Movie ID, (Predicted Rating, number of ratings))
predictedWithCountsRDD  = (predictedRDD
                           .join(movieCountsRDD))
# Use RDD transformations with PredictedWithCountsRDD and moviesRDD to yield an RDD with tuples of the form (Predicted Rating, Movie Name, number of ratings), for movies with more than 75 ratings
ratingsWithNamesRDD = (predictedWithCountsRDD
                       .join(moviesRDD)
                       .map(lambda s:(s[1][0][0],s[1][1],s[1][0][1]))
                       .filter(lambda s:s[2] > 200)
                       .map(lambda s:(s[0],s[1])))
predictedHighestRatedMovies = ratingsWithNamesRDD.takeOrdered(20, key=lambda x: -x[0])
print ('My twenty highest rated movies as predicted (for movies with more than 200 reviews):\n%s' %
        '\n'.join(map(str, predictedHighestRatedMovies)))

