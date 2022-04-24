# DSAI Mini Project
CZ1115 - Introduction to Data Science and Artificial Intelligence Mini Project

# Welcome to our mini project repository

## About

This is a mini project for CZ1115 which focuses on movies from [IMDB 5000 Movie Dataset](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset). 

In particular, our notebook details the following:
1. Data Processing (Handling Null Values and Removing Duplicates)
2. Feature Engineering
3. EDA for Response Variable, Gross
4. EDA for Numeric Features
5. EDA for Categorical Features
6. Dimensionality Reduction and Robust Scale Transformation
7. Baseline Models
8. Improvements to Baseline Models
9. Model Discussions and Conclusion

## Problem Definition
Every year thousands of movies are filmed and produced but not all of them breakeven. An even lesser number of movies make it as a "box-office hit". In this project, we intend to create a machine learning model that predicts which movies generate more gross revenue based on information available on IMDB, an online database for movies 

## Models Used

1. Linear Regression
2. Decision Trees
3. Random Forest
4. Gradient Boosting

## Conclusion
1. Decision Tree Model's poor performance
- Decision Trees have a high tendency to overfit the training set if there is no stopping criteria.
- Due to overfitting, the model will not perform well without hyperparameter turning which forces the tree to terminate early

2. Why is Gradient Boosting the better tree-based model in theory
- Decision Tree, Random Forest and Gradient Boosting are tree-based models. Random Forest and Gradient Boosting both use Decision Trees as base models.
- Random Forest is more accurate than a single decision trees as it's an ensemble learning technique that employs bagging (bootstrapping and aggregating) - it constructs a set of base models from the training set and make predictions by aggregating the predictions made by each base model. RF is more accurate than a single Decision Tree as it incorporates more diversity.
- Gradient boosting uses boosting on top of bagging used in Random Forest and performs better as it boosts a set of weak learners (decision trees called stumps) to strong learners such that misclassifications made by weak learners are taken into consideration. In general, the distribution of training set is changed adaptively so that weak learners will focus more on errors made by previous learners. Decision Trees are used as the weak learners and are added one at a time, and fitted to correct the errors made by previous trees. The errors made can be captured by a cost function, which is to be minimised using the gradient descent algorithm (hence the name gradient boosting). Thus, Gradient Boosting is the better tree-based model.

3. Best model to predict gross revenue for Top IMDB Movies dataset
- From a theoretical standpoint, Gradient Boosting should perform better than Decision Tree and Random Forest due to the above reasons. It should also perform better than the basic linear regressor. Indeed, from our results, Gradient Boosting performed the best among the 4 models for all variations (baseline, outliers removal, feature selection, hyperparameter tuning).
- From our results, it is safe to say that Gradient Boosting is the best tree-based model to predict the gross revenue of top IMDB movies from this dataset.
- However, as there are other strong models that were not covered in the scope of this project (eg. Multi-Layer Perceptron, Support Vector Machine), we cannot be certain that Gradient Boosting is the best model in predicting gross revenue for top IMDB movies.

4. Top features to predict gross revenue for Top IMDB Movies Dataset
Based on the feature importance ranking, the top 5 features are:
  - Number of users who voted
  - Budget
  - Number of critic reviews
  - Whether the movie is of the genre 'Family'
  - IMDB score
However, it is not clear how these features actually affect the gross revenue. We only know that they are important in terms of reducing the impurity for our Gradient Boosting Regressor.
Out of the top 3 features which had the highest correlation with the target variable gross from the correlation matrix, 2 of them appeared in the top 5 features (Number of votes by users and Number of critic reviews).

5. Conclusion
- Every step of the EDA is vital in ensuring that our model inputs are not nonsensical. This ensures that our model predictions are meaningful (Garbage In, Garbage Out principle).
- The dataset has to be processed properly, such as the handling null values and removing of duplicates. As the feature set is small, more features need to be engineered and transformed into sensible features (eg. using one-hot encoding).
- Plotting the feature distributions allows us to identify skewness and outliers. Appropriate techniques (eg. robust scale transformation) can then be used to resolve them.
- By understanding the underlying algorithms behind the models, we can expect certain results and improve our model performance (eg. hyperparameter tuning). From our findings, we are able to determine the model (Gradient Boosting) that best predicts the gross revenue of top IMDB movies from our dataset and the important features that are used by the model to make predictions.

## What did we learn from this project?
On top of what was taught in the CZ1115 curriculum, here's what we learnt along the way:
- Data preprocessing
  - Handling null values
  - Preparing data for ML model via Dimensionality Reduction (to combat high cardinality features and the curse of dimensionality) and Robust Scaler Transformation using sklearn (to combat outliers) 
- Feature engineering
  - Adding new features based on existing information in dataset
  - Manipulating difficult features such as genre and plot_keywords with string values and coercing them into numerical values with one hot encoding so that our ML model is able to process the data in a meaningful manner 
- Random Forest using sklearn
- Gradient Boosting using sklearn
- Feature importance using Mean Decrease in Impurity using sklearn
- Hyperparameter turning and k-fold cross validation with GridSearchCV from sklearn
- Loss functions, gradient descent
- Concepts on ensemble machine learning techniques (bagging and boosting)
- Random Forest, Gradient Boosting and their performance

## Contributors

- Truong Quang Duc @Mine-Power - Data Preprocessing, Feature Engineering, Random Forest
- Riley Ang @rileyang1 - EDA, Dimensionality Reduction and Robust Scale Transformation, Gradient Boosting
- Jordon Kho @jaykjy - Linear Regression, Decision Trees, Discussions and Conclusions

## References

- https://machinelearningmastery.com/dimensionality-reduction-for-machine-learning/
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
- https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
- https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
- https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502
- https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/
- https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
- 
