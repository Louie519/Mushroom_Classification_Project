# Mushroom Classification: Edible vs. Poisonous

This project classifies mushrooms as either edible or poisonous based on various physical characteristics using machine learning. The primary model used is the RandomForestClassifier, along with PCA and K-Means clustering for unsupervised learning analysis.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
  - [Random Forest Classifier](#random-forest-classifier)
  - [PCA and K-Means Clustering](#pca-and-k-means-clustering)
- [Model Evaluation and Scores](#model-evaluation-and-scores)
- [Feature Importance](#feature-importance)
- [PCA and K-Means Clustering Results](#pca-and-k-means-clustering-results)


## Project Overview

The goal of this project is to classify mushrooms as either edible or poisonous based on various features like odor, cap shape, and habitat. The project uses a RandomForestClassifier to predict edibility, with additional insights gathered using PCA and K-Means clustering.

## Dataset Information

The dataset contains the following features:

- **Cap Shape, Cap Surface, Cap Color**
- **Bruises (Yes/No)**
- **Odor (almond, creosote, fishy, foul, etc.)**
- **Gill Size, Gill Color**
- **Habitat (woods, urban, meadows, etc.)**

The target variable is `edible`, a binary label indicating whether a mushroom is edible (`1`) or poisonous (`0`).

## Data Preprocessing

1. **Data Cleaning**
   - Dropped the `veil-type` column due to a single unique value.
   - Removed missing values (`?`) from the `stalk-root` column.
   - Created a binary `edible` column for the target classification.

2. **Feature Engineering**
   - Converted categorical variables into dummy/one-hot encoded variables.
   - Combined the encoded variables with the main dataset.

3. **Dimensionality Reduction**
   - Applied Principal Component Analysis (PCA) to reduce feature dimensions while retaining most of the variance.

4. **Clustering**
   - Implemented K-Means clustering to explore underlying patterns in the data.

## Modeling

### Random Forest Classifier

We used a RandomForestClassifier to classify mushrooms as either edible or poisonous based on the preprocessed dataset. The data was split into training and testing sets, with 80% used for training and 20% for testing.

### PCA and K-Means Clustering

**Principal Component Analysis (PCA)** was used to reduce the number of features for visualization purposes, and **K-Means clustering** was applied to group mushrooms into clusters.

- **PCA**: Reduced the dataset into two principal components for visualization.
- **K-Means**: Applied K-Means clustering with three clusters to analyze the data.

#### PCA Steps:
1. **PCA with Two Components**: Reduced the feature space into two components while retaining most of the variance for visualization.
2. **Cumulative Explained Variance**: Showed how much variance each principal component captured.
3. **Optimal Number of Components**: 15 components explain 90% of variance. 64% of the variance was explained by the first two components.

#### K-Means Clustering:
1. **K-Means**: Performed K-Means clustering with k=3 clusters.
2. **Elbow Method**: Used to find the optimal number of clusters.
3. **Silhouette Scores**: Used to evaluate the quality of clustering.

## Model Evaluation and Scores

### Random Forest Classifier

The Random Forest model performed very well in classifying mushrooms based on the features.

- **Accuracy**: 100%
- **ROC AUC Score**: 0.99

### PCA and K-Means Clustering Results

The PCA and K-Means clustering demonstrated the following key insights:

- **PCA** reduced the dataset into two principal components.
- **K-Means Clustering** showed clear and distinct clusters when k=3.
- **Silhouette Scores** for clustering indicated that the mushrooms formed very distinct groups.

- **Silhouette Score for K-Means**: 0.80

## Feature Importance

The top 5 most important features from the RandomForestClassifier were:

1. **Odor**
2. **Spore Print Color**
3. **Gill Size**
4. **Population**
5. **Habitat**

## Random Forest Model Classification Report

|               | precision | recall | f1-score | support |
|---------------|------------|--------|----------|---------|
| Poisonous (0)    | 1.00       | 1.00   | 1.00     | 163     |
| Edible (1) | 1.00       | 1.00   | 1.00     | 184     |
| **Accuracy**  |            |        | 1.00     | 347     |
| **Macro avg** | 1.00       | 1.00   | 1.00     | 347     |
| **Weighted avg** | 1.00    | 1.00   | 1.00     | 347     |

