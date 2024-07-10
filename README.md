# Cherry Leaves Mildew Detection

## Introduction

The Cherry Leaves Mildew Detection app uses Machine Learning to allow users to upload images of leaves, identify whether a leaf is healthy or infected with powdery mildew and see detailed ML Performance Metrics.

## Table of Contents

## Business Requirements

The cherry plantation crop from Cherries & Berries is facing a challenge where their cherry plantations have been presenting powdery mildew.Currently, the process involves manually verifying whether a given cherry tree has powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute.  The company has thousands of cherry trees, located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual inspection process.

To save time in this process, the IT team suggested an ML system that detects instantly, using an image of a cherry tree leaf, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Cherries & Berries, taken from their crops.


* 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
* 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Hypothesis and how to Validate

Implementing a machine learning model trained on cherry leaf images will improve the accuracy and efficiency of detecting powdery mildew compared to traditional manual diagnosis methods.

* Cherry leaves with powdery mildew can be distinguished from healthy ones by their distinctive whitish appearance.
    * This can be verified by creating an average image study and image montage to determine differences in the appearance of healthy leaves and leaves with powdery mildew.
* Cherry leaves can be determined to be healthy or infected with a degree of 99% accuracy.
    * This can be verified by evaluating the model on the test dataset, which should achieve at least 99% accuracy.

## Rationale to Map the Business Requirements to the Data Visualizations and ML Tasks

### Business Requirement 1: Data Visualization

* Visualizing 'Mean' and 'Standard Deviation' for Healthy and Infected Cherry Leaves
* Visual Distinction between Typical Healthy and Infected Leaves
* Collection of Images Showcasing Both Healthy and Infected Leaves

### Business Requirement 2: Classification

* Building and training a machine learning model to classify cherry leaves as either healthy or infected with powdery mildew.
* The predictions should have a 97% accuracy level.

## ML Business Case

The Cherry Leaf Mildew Detection project aims to leverage machine learning (ML) to enhance the efficiency and accuracy of identifying powdery mildew on cherry leaves. Currently, the process relies heavily on manual inspection, which is time-consuming and prone to human error. By implementing an ML model trained on cherry leaf images, the project seeks to automate and streamline the detection of powdery mildew.

* Objectives:
    * Enhanced Diagnostic Accuracy: Improve the ability to detect powdery mildew on cherry leaves, reducing the chances of missed infections and incorrect diagnoses.
    * Improved Efficiency: Expedite the detection process, enabling farmers and agricultural professionals to make timely decisions to manage and prevent mildew outbreaks.
    * Cost Efficiency: Decrease operational costs associated with manual inspection and potential re-evaluations of leaf health.

* Value Proposition:
    * Accurate and Timely Detection: Provide agricultural professionals with a reliable tool for quickly identifying powdery mildew on cherry leaves, leading to effective and timely treatment.
    * Operational Efficiency: Streamline inspection processes by reducing manual effort and improving resource allocation in managing cherry crops.
    * Cost Savings: Reduce the expenses related to prolonged inspections and the risks of inaccurate diagnoses.

* Implementation Strategy:
    * Data Collection and Annotation: Gather a diverse dataset of cherry leaf images annotated for the presence of powdery mildew. The dataset can be found on Kaggle. The dataset contains images of healthy cherry leaves and leaves affected by powdery mildew.
    * Model Development: Develop and optimize an ML algorithm capable of accurately detecting and classifying cherry leaves as either healthy or infected with powdery mildew.
    * Validation and Testing: Validate the model's performance using testing protocols. The model will be deemed successful if it achieves an accuracy of at least 97% on the test set.
    * Deployment and Integration: Integrate the ML model into a user-friendly web application or dashboard to support seamless adoption by farmers and agricultural professionals.
    * Monitoring and Iteration: Continuously monitor the model's performance, gather feedback from users, and iterate to improve accuracy and reliability.

The Cherry Leaf Mildew Detection project represents a strategic initiative to harness ML technology for improving agricultural outcomes. By automating the detection of powdery mildew, the project aims to enhance diagnostic accuracy, operational efficiency, and overall crop management while reducing costs associated with manual inspection processes.

# User Stories

As a Farmer or Agricultural Professional:
* I want to use a machine learning app to assist in identifying powdery mildew on cherry leaves, so that I can improve the accuracy and speed of my disease diagnosis and management.
* I want a straightforward and user-friendly dashboard app, so that I can easily navigate the tool without needing extensive training or technical expertise.
* I want to examine average and variability images of both healthy leaves and leaves with powdery mildew, so that I can visually differentiate between these two classifications and understand the visual characteristics of the disease.
* I want to observe the disparity between an average healthy cherry leaf and an average leaf affected by powdery mildew, so that I can clearly see the differences and better identify the disease in my crops.
* I want to review an image montage displaying both healthy leaves and leaves with powdery mildew, so that I can visually discern the differences between the two conditions and identify them in the field.
* I seek to upload cherry leaf images and receive classifications with over 97% accuracy, so that I can quickly and reliably determine whether the leaves are healthy or infected with powdery mildew.

# Ethical and Privacy Concerns

The cherry leaf mildew dataset used in this project was provided by the client under a strict Non-Disclosure Agreement (NDA). This agreement ensures that the data is used exclusively for the purposes of this project and is not shared outside the scope of the defined collaboration.

* The dataset is only accessible to authorized personnel directly involved in the project. Access is strictly controlled to ensure that only the team members who are officially part of the project have the ability to view and work with the data.

*  All individuals involved in the project have signed a confidentiality agreement, committing to maintaining the privacy and security of the data. The terms of this agreement prohibit sharing, distributing, or using the data beyond the agreed-upon project scope.

* We adhere to ethical guidelines in the use of the data, ensuring that it is used solely for the development and improvement of the machine learning model for cherry leaf mildew detection. Any findings or results from the data are reported in a manner that respects the privacy and confidentiality of the information.

* The data will not be shared with any external parties or organizations outside the official project team. The project’s goals and outcomes are communicated in a way that respects the NDA and maintains the confidentiality of the data.

# Methodology

The CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology was pivotal in guiding this Cherry Leaf Mildew Detection project through its phases:

Business Understanding:
* Defined the project objectives to improve the accuracy and efficiency of detecting powdery mildew on cherry leaves using machine learning.
* Identified stakeholders' needs, seeking faster and more reliable disease detection to enhance crop management and reduce losses.

Data Understanding:
* Gathered diverse datasets of cherry leaf images, including both healthy leaves and leaves affected by powdery mildew.
* Conducted exploratory data analysis (EDA) to understand the data distributions and characteristics, identifying key features for disease detection.

Data Preparation:
* Cleaned and preprocessed the data to ensure consistency and quality, including handling missing values, normalizing image sizes, and augmenting the dataset to enhance model robustness.
* Extracted relevant features from the images and formatted them for machine learning model input, ensuring the data is suitable for training and evaluation.

Modeling:
* Selected appropriate machine learning algorithms for categorical classification of healthy versus infected leaves.
* Trained multiple models using the prepared data, experimenting with different algorithms and hyperparameters to identify the best-performing model.

Evaluation:
* Evaluated model performance using metrics such as accuracy and loss.
* Used cross-validation techniques to assess model robustness and generalization to new data, ensuring the model performs well on unseen images.

Deployment:
* Integrated the best-performing model into a user-friendly application, providing an intuitive interface for farmers and agricultural professionals.
* Ensured a working Heroku application for seamless adoption by users, allowing them to upload images, receive accurate disease classifications, and download reports with predictions.

# Rationale for the Model

An effective machine learning model makes accurate predictions by generalizing well from the training data to new, unseen data. Ideally, such a model should also be as simple as possible, avoiding unnecessary complexity in its neural network and high computational demands.

If a model is trained for too long on the training dataset or is overly complex, it might start learning the noise or irrelevant details from the data. This leads to overfitting, where the model performs exceptionally well on the training data but poorly on new data due to its inability to generalize. Overfitting can be identified by evaluating the model's performance on validation and test datasets.

Underfitting happens when the model fails to capture the underlying relationship between the input and output data. This can be detected by its poor performance on the training dataset, which usually results in similarly low accuracy on the validation and test datasets.

# Model Creation

This project focuses on image classification, requiring the implementation of a Convolutional Neural Network using TensorFlow. The goal is to develop a model for categorical image classification, distinguishing between healthy and infected outcomes.

For categorical classification tasks, the approach involves choosing between two options: using a single neuron with a sigmoid activation function or employing two neurons with a softmax activation function. Both configurations were tested and fine-tuned during the experimental phase.

The final model is a sequential model that includes the following components:

* Convolutional Layers: Used to identify patterns within the image by applying filters that distinguish dominant pixel values from non-dominant ones.

* The model includes 3 convolutional layers.

* Conv2D was used as the images are in 2D.

The number of filters used were 32, 64, and 64, progressively increasing to capture more complex features.

* The kernel size of 3x3 was chosen for its efficiency.

* The activation function ReLU was used for its simplicity and effectiveness in hidden layers.

Pooling Layers: Applied following each convolutional layer to reduce the image size by capturing the most significant pixels.

* MaxPooling2D was used to emphasize the brighter pixels in the image, which helps in simplifying the image complexity.
* Flatten Layer: Converts the matrix into a vector, creating a unified list of all values to be passed into a dense layer.

Dense Layer: A densely connected neural network layer.

* 128 nodes were chosen through a process of trial and error.
* The activation function ReLU was used.
* Dropout Layer: A regularization layer implemented to mitigate the risk of overfitting in the neural network.

* A dropout rate of 0.5 was chosen.

Output Layer:

* The sigmoid activation function was determined to be optimal for binary classification. This configuration uses 1 node to provide a single output value representing the probability of the positive class (infected).
The binary_crossentropy loss function was selected, paired with the adam optimizer, chosen after thorough experimentation and testing.





