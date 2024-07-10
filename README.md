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