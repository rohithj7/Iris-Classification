# Iris Classification

## Report

The Iris dataset comprises 150 observations of iris flowers from three different species: Iris setosa, Iris virginica, and Iris versicolor. Each species has 50 observations, making the dataset balanced in terms of class distribution.

Data Attributes:
The dataset consists of four features measured in centimeters for each sample:
- Sepal Length: The length of the sepal (the outer parts of the flower that protect the petals before it blooms).
- Sepal Width: The width of the sepal.
- Petal Length: The length of the petal (the colorful parts of the flower).
- Petal Width: The width of the petal.
These measurements are used to predict the species of the iris flower, which is the categorical target variable.

The scatterplot of the dataset looks like this:
![Screenshot 2024-09-14 000116](https://github.com/user-attachments/assets/92e9aca9-c157-4dd0-86cc-147dce338ff8)
This plot tells us that class 0 is significantly far away from classes 1 and 2 as the petal lengths are clearly on the lower end of the scale. However, for classes 1 and 2, as the petal lengths overlap for some flowers. As a result, it’s much easier to classify a class 0 flower than it is the classes 1 and 2 from some petal lengths.

Classification Report:
![Screenshot 2024-09-14 000246](https://github.com/user-attachments/assets/5b4e890b-3683-482d-8581-fc0efdd65a89)

I chose my hyperparameters for all the algorithms by using GridSearchCV by using a model along with the parameter grid comprising a list of the possible values for each hyperparameter. GridSearchCV gave me the best/optimized values for the hyperparameters for each of the algorithms.

Naive Bayes provides robust results given its assumption of feature independence. Its high ROC AUC score suggests excellent capability in distinguishing between the iris classes. SVM leads in all three metrics, indicating its high efficiency in managing the iris dataset due to its kernel trick. Random Forest, an ensemble method, shows strong performance and generalization but slightly trails SVM. It is less likely to overfit compared to other models, given its nature of averaging multiple decision trees. KNN performs well, especially given that the iris dataset is fairly small. xGBoost, another ensemble method leveraging gradient boosting, seems to perform less optimally compared to other models.

When receiving the classification report, I did not feel the need for any data preprocessing considering the models, especially SVM and KNN classification which necessitate data preprocessing in some cases, were performing well when evaluated with 5-fold cross validation. However, overfitting might be a concern, especially for high-complexity models like xGBoost. Nevertheless, we utilize 5-fold cross validation to ensure we are reducing the possibility of overfitting.

Overall, the SVM shows the best performance across all metrics, making it a suitable choice for the iris dataset. The high ROC AUC scores across the models indicate good classifier separability for the different iris classes.
