# New York AirBnB Price Predictions
Since the number of listings hit their peak just before the pandemic, prospective hosts have been seeing fewer and fewer bookings as the market has been oversaturated. Combined with the slowing economy, hosts have had a more difficult time pricing their units. Our model aims to help inform these homeowners’ decisions by analyzing the overall market and deciding what a fair price for their unit would be to base their choice off of. 

---

For this project, we used the `New York Airbnb Open Data 2024` dataset by Vrinda Kallu, available [here](https://www.kaggle.com/datasets/vrindakallu/new-york-dataset/data).

We began our data exploration by loading both a pairlot and a heatmap for our data to determine any correlations amongst the data. By viewing the data in these forms we could clearly see there appear to be a few outliers on the plots for price where we can clearly see a single point distant from the rest of the data. Also, we looked at the number of null values, unique columns, and the data types for each column before conducting data cleaning.

After conducting our initial data exploration, we first decided to drop some features that are unrelated to our question, such as `id`, `name`, `host_id`, `host_name`, `latitude`, and `longitude`. We found quite a few of the features are object values as they are mixtures of strings and numerical values in the features: ratings and baths. In these features we found string values indicating these listings had no values there. We handled this by dropping the not specified values for the baths as throughout exploration we found there were only 13 listings where the number of baths was marked as `Not Specified`. On the other hand we found that 1815 of the listings had been marked as having `No Rating` or as `New` indicating it has no rating yet. Thus, we decided to perform median imputation instead of mean imputation since there was an extremely large value of outlier. Additionally to handle the categorical features we decided to one hot encode room_type and neighbourhood_group in our initial preprocessing. We also found through our exploration that the `bedrooms` feature has a mixture of numerical values and `Studio`. In order to handle this data we set all instances of `Studio` to 0 in the `bedrooms` column and made a one hot encoding column for the `Studio` values. At some point, we plan on normalizing and standardizing the data to see the importance of predicted values using weights by making it easier to compare and analyze.

## [Model 1: Linear Regression](https://colab.research.google.com/drive/1jjwC8OQ4t2foMpVOL1rNyTsa6Zegyr1H#scrollTo=x0l9IybfOdcb )
Looking at the predictive error for our linear regression model, the training, validation, and testing MSE were all very high (10760, 9347, 10208). This tells us that in the fitting graph, our model fits on the leftmost side where we have a very simple model that is underfitting the data. We also experimented with a [polynomial regression](https://colab.research.google.com/drive/1jjwC8OQ4t2foMpVOL1rNyTsa6Zegyr1H#scrollTo=vlLf2gjQPPag) model with a degree of 4, yet our MSE was still high for both the training and testing MSE, with the testing MSE being significantly higher than the training (7350 versus 1.74e22). This model would fit on the right side of the fitting graph (overfitting), despite the high training MSE. 

For our first model we choose to use Linear Regression. Our linear regression model performed poorly. With it being our most simple model, there is little we can do to improve the mean squared error (MSE) of the model. After initially running the model and seeing an extremely high MSE, we reevaluated our feature data to filter any outliers, which were any AirBNB listings with a minimum price over $1000 per night. After rerunning the linear regression model on the data excluding the outliers, the MSE improved although was still performing quite poorly. This seems to suggest that there are little to no linear relationships between any of the numerical features and price. To improve this model, we could plot some of the feature distributions to see if there are outliers in them and drop these rows as well. However, since we have found there is not a linear relationship between the features we selected and our target `price`, the best step moving forward would be to test out an entirely different model. 

One possible model we can implement next is a neural network. Trying several hidden layers with different numbers of units (nodes) per layer and different kinds of activation functions would help find the best model for predicting the price, which would be more  accurate than the current linear, polynomial, or logistic regression models. 
Another model that we can implement is a support vector machine. We observed that our MSE was extremely high for a linear regression model which is a relatively simple model. A more complex model like SVM will try to find a hyperplane to model the distribution which could potentially be more accurate than a simple linear regression.

## [Model 2: Sequential Neural Network](https://colab.research.google.com/drive/1jjwC8OQ4t2foMpVOL1rNyTsa6Zegyr1H#scrollTo=qA7ON3QSub6I)
For our second model we chose to use a Sequential Neural Network. For this model, our existing dataframe and columns were sufficient, and we are continuing to use mean squared error as our loss function. Similar to our first model, our second model did not perform as well with a MSE for our training, testing, and validation data being: 10512, 11065, 10786. With those MSE's in mind, we observe our model is fairly consistent in performance and is on the left side of the fitting graph similar to our first model. This means our model was still fairly simple and underfitted our data.

In this model, we did not perform any hyper parameter tuning and any feature expansion. However, we did have K-fold cross validation, with 10 folds repeated 5 times. The results were still similar to our first model in that the MSE was roughly around 10000.

One model that we considered using is Random Forest Regressor because our data has a lot of dimensionality and the relationship between our feature variables and our target variable is not linear. Random Forest Regressors are much more complex than a simple model like Linear Regression and our Sequential Neural Network and are more likely to handle our complex data better.

Our second model, the Sequential Neural Network performed similarly to our first model, Linear Regression. This is likely due to the simplicity of the model as although our model had more depth and layers to it, it is fairly simple compared to other neural network models. Somethings that can be done to possibly improve our model is to do some feature expansion and hyper parameter tuning as there were many activation functions and layers that we could have experimented with.

## [Model 3: Random Forest Regression](https://colab.research.google.com/drive/1jjwC8OQ4t2foMpVOL1rNyTsa6Zegyr1H#scrollTo=GV66h11LR_tT)
For our final model we decided to use a Random Forest Regressor. Similar to the other models, we did not need to alter our data further and continued using the same dataframe and loss function (MSE). Unlike our first two models, our third model performed better than those two in training, testing, and validation MSE, with a score of 1510, 8158, 9327. We observe that our model is overfitting because the training score is much lower compared to the testing score and validation score. In other words, our model would lie on the right side of the fitting graph, being a complex and overfitted graph.

In this model, we did not perform any hyper parameter tuning and any feature expansion. However, we did have K-fold cross validaiton, with 10 folds repeated 5 times. The results were 9327.

Our third model, the Random Forest Regression performed much better compared to our first two models, Sequential Neural Network and Linear Regression. This is likely due to how the model is much more complex than our first two models and thus able to better handle our data. Somethings that can be done to possibly improve our model is to do some feature expansion and hyper parameter tuning in order to make our model overfit our data less and make the scores between training, testing, and validation more consistent and closer to 0.

### Final Notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jjwC8OQ4t2foMpVOL1rNyTsa6Zegyr1H?usp=sharing)

