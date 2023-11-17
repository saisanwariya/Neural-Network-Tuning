<h1 align="center">Neural Network Tuning and Visualization for MNIST Digit Recognition</h1>

## Overview
This project, developed by Sai Sanwariya Narayan, focuses on enhancing machine learning performance in digit recognition using the MNIST dataset. It leverages cluster computing and advanced analysis techniques for optimal tuning and visualization of neural networks. The program employs PySpark for distributed computing and sklearn for initial data handling, culminating in a highly efficient classifier capable of identifying handwritten digits with high accuracy.

## Program Functionality
1. **Data Preprocessing:** Utilizes pandas and sklearn to download and preprocess the MNIST dataset, transforming it into a suitable format for machine learning.
2. **Spark Session Creation:** Initiates a Spark session for distributed computing.
3. **Feature Engineering:** Employs VectorAssembler and StandardScaler from PySpark for feature vector creation and scaling.
4. **Neural Network Modeling:** Constructs and trains a Multilayer Perceptron Classifier, defining layers and utilizing hyperparameter tuning for optimization.
5. **Model Evaluation:** Uses the MulticlassClassificationEvaluator for assessing the model's performance based on the F1 score.
6. **Model Saving and Loading:** Includes functionality to save the best-performing model and load it for future predictions.
7. **Prediction Visualization:** Implements matplotlib-based visualization to display classifier predictions alongside the corresponding MNIST images.


## Notes
- The program is designed to run in a cluster computing environment, ensuring efficient processing of large datasets.
- Users looking to replicate or modify the project should have a basic understanding of Python, machine learning, and Spark.
- The visualization component is not configured for cluster mode and is intended for local execution.

---

# Academic Integrity Statement

Please note that all work included in this project is the original work of the author, and any external sources or references have been properly cited and credited. It is strictly prohibited to copy, reproduce, or use any part of this work without permission from the author.

If you choose to use any part of this work as a reference or resource, you are responsible for ensuring that you do not plagiarize or violate any academic integrity policies or guidelines. The author of this work cannot be held liable for any legal or academic consequences resulting from the misuse or misappropriation of this work.

Any unauthorized copying or use of this work may result in serious consequences, including but not limited to academic penalties, legal action, and damage to personal and professional reputation. Therefore, please use this work only as a reference and always ensure that you properly cite and attribute any sources or references used.
