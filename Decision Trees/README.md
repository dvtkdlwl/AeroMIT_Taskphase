# Decision Trees Report

A decision tree is a popular machine learning model that can be used for
both classification and regression. It has multiple nodes- and each of
these nodes serves a different purpose. The root node is the starting of
the decision tree, and it is where a testing datapoint first enters the
decision tree to be classified. Multiple questions are asked about the
point at every internal node at each level of the decision tree, and on
the basis of each of these questions, the point finally gets classified
into one category of output label. 
<p align = "center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/4b182f81-a68a-4082-8e27-15e1e8a17ea0" />
</p>

The decision tree is called a binary classification tree if there are
only two possible output labels, while it is called a multiple
classification tree if there are more than 2 possible output target
labels.

I will first talk about classification trees, and later cover regression trees.

## Classification Trees

During training of the classification tree, at the root node, we receive
the entire training dataset. Our goal is to divide up these datapoints
on any one of the features. We therefore iterate through all the values
of all the columns and calculate at which threshold of what column, we
receive the best possible split. The objective parameter for calculating
this is the Information Gain (calculated by the Entropy/Gini Index in
our case). Once we have the best threshold for the best feature, we
apply an argument to split the data points on whether this feature of
theirs is larger than or smaller than our decided threshold. In this
way, we have separated the entire input training dataset into two sets,
while at the same time establishing splitting conditions for the same.
We apply this very logic to also split the dataset further at child
nodes. At a pre-determined hyperparameter max_depth (say 10 levels), the
splitting can be stopped and the decision tree can then be used for
testing.

For testing, I used the F1 score.

F1 score is given by **(2 \* Precision \* Recall) / (Precision +
Recall)  
Precision = TP / (TP + FP):** Where precision tests how many of the
model's predicted True values were actually True with respect to its own
guesses.**  
Recall = TP / (TP + FN):** Where recall checks how well the model is
able to identify all the True values with respect to the original
dataset.

Another thing to note is that F1 score usually gives a number, like the
R2 score. If we want a proper yes/no answer on the basis of an entire
testing set, then we can use techniques to aggregate the output data,
such as majority voting (for classification) or averaging (for
regression).

## Regression Trees

While classification trees predict categorical values, regression trees
predict continuous numerical values e.g. the cost of a house based on 10
different features.

For regression trees, even though the rough structure of the decision
tree is the same, the best-split criteria is a little different. Where
we used Gini Index/ Entropy to calculate the impurity of a node, here we
use variance (as all the values are numeric). The main goal therefore
becomes to reduce variance as much as possible with every single split.

While training, the variance of the root node (all target labels in the
training dataset) is calculated. And then the best possible split is
found such that the column and splitting threshold minimise this
variance for child nodes. This process repeats till a pre-determined
depth, or until the model has started creating too many leaf nodes with
too little datapoints in each leaf node (classic sign of overfitting).
The values of the leaf nodes are then calculated on the basis of the
mean/median of all values in the leaf node.

After training, the model retains all these parameters. And then, the
testing dataset is traversed through this established decision tree in
order to get an output.

## Overfitting- A Common Problem with Decision Trees

Decision trees frequently learn the training data too well, and this
means that the tree has overfit- i.e. it is not able to generalise well
and predict test data accurately. To avoid overfitting, a variety of
techniques can be used:

1)  *Maximum Depth:* As seen above, we can easily set a maximum depth
    hyperparameter on our decision tree in order to prevent it from
    overfitting. What this does is that it limits the decision trees
    from creating too many levels and asking too many questions to the
    datapoint, ultimately allowing somewhat error to creep into our
    training data accuracy for the sake of better generalisation towards
    test data.

2)  *Pruning:* Pruning basically refers to the cutting off of a branch
    of the decision tree and it is of two types: pre-pruning and
    post-pruning:

<!-- -->

a)  Pre-Pruning- Pruning which is done while the tree is being built
    (trained). One example is to set the minimum samples per leaf, which
    avoids the decision tree from creating a lot of leaf nodes for just
    1-1 point each. Another can be to set a minimum number of samples
    per split, which again sets a lower bound and prevents splitting
    scanty datasets. We can also set a maximum number of features to be
    taken into consideration (especially handy in case of random
    forests). This is good because some features are very powerful at
    creating good splits, while others may not be.

b)  *Post-Pruning-* After the­ tree is fully grown, post-pruning involves
    re­moving branches or nodes to improve the­ model\'s ability to
    generalize­. An example is Cost-Complexity Pruning (CCP), which
    assigns a price to each subtre­e based on its accuracy and
    complexity, the­n selects the subtre­e with the lowest fee. Another is
    that of Re­duced Error Pruning, which removes branche­s that do not
    significantly affect the overall accuracy. Finally, the Minimum
    Impurity De­crease pruning technique Prunes node­s if the decrease­ in
    impurity (Gini impurity or entropy) is beneath a ce­rtain threshold
    and therefore is not lucrative enough to create.

## Sampling Techniques- How to Deal With Imbalanced Data?

We come across very imbalanced data frequently. Imbalanced data refers
to data that has a majority and minority class- i.e. one target label is
present in extremely high quantities when compared to the other target
label. For example, in disease identification where 97% people getting
tested do not have a disease and only 3% do. Or in another case, credit
card defaults- where only about 5% of customers actually default while
95% do not.

An issue can arise due to this in our model. If we use imbalanced data
to train any supervised machine learning model, the model might simply
fail to learn the minority class well. This is an issue that might not
even be immediately noticed- because the model will have 95% accuracy
even if it predicts the majority class 100% of the time!

This is where the concept of sampling comes into play. Sampling is a
technique that is used to deal with imbalance in data sets, and it is of
two types: Undersampling and Oversampling.

Now, there are multiple oversampling and undersampling techniques we can
use if we have a class imbalance. For example, let\'s say that our data
is binary-classified. Then, having 80% samples as Type-0 and 20% samples
as Type-1 would mean that a model trained on this data would have an
inherently better understanding of Type-0, while it may be weaker at
predicting Type-1.  
Then we have two possible sampling techniques to deal with this:

1.  Undersampling- Eliminates the number of points of majority class
    randomly to enforce an equal class balance. However, this is not
    very compelling because it reduces the dataset without adding too
    much real value. Moreover, it could increase the sampling bias into
    our model- ie the model\'s parameters keep varying for different
    randomly-selected sample datasets that are picked.

2.  Oversampling- Generates synthetic points on the dataset based on
    currently existing points, and sometimes even creates more points
    from these newly-generated synthetic points.

a\. The most basic way to oversample is to duplicate existing data
points up to a certain amount. Even though it reduces training bias
towards majority class, this method can prove to be too basic and lead
to overfitting. Training it is also resource-intensive because the size
of the training set has now increased drastically.

b\. Another example of this technique is SMOTE, Synthetic Minority
Oversampling Technique. SMOTE basically finds a pre-determined (e.g. 5)
neighbours of all points in the minority class. Then, it draws a line
between each existing point and any one of its randomly selected
neighbours. On this line, SMOTE adds a new point (this could be in the
exact centre between the two points, or it could be a randomly-found
distance from either point). This fills up the sample space for Type-1
as well. However, SMOTE also has its problems. For example, it can be
slow to train as it uses the KNN algorithm which is very
resource-expensive. SMOTE does not work as well for categorical data
(e.g. a bar graph of 0, 1, 2, 3 could generate a synthetic point between
2 & 3- which is simply not possible!)

c\. The third example is ADASYN- which is Adaptive Synthetic Sampling.
It basically interpolates the minority points nearer to the boundary as
they are harder to classify normally. Adding more points to this
boundary increases the dataset and hence understanding of these
borderline points near the boundary.
