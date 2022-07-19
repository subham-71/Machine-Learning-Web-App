# Data Pre-processing and Model Selection Web App
A web app to simplify the data pre processing steps and hyper parameter tuning process of machine learning models. It helps in selecting the best model with optimal parameters.
The web app has services for both data pre-processing steps and machine learning model hypertuning process.

## File Upload
First a file is uploaded to the web-app. The file for pre-processing can be raw. But for building a machine learning model and selecting the best model with optiimal parameters, a pre-processed file needs to be uploaded with the target feature at its last column. It can be either a regression or a classification task.

File upload format : cdv,txt,xls,xlsx,ods,odt<br />
I have also given link to download sample datasets so as to explore the app.<br />

#### Sample Datasets :

sample_file_for_preprocessing    : You can use this file for pre-processing tasks <br />
regression_preprocessed_file     : You can use this file to build regression models <br />
classification_preprocessed_file : You can use this file to build classification models <br />


![front](https://user-images.githubusercontent.com/72215169/171049013-9d06a1fd-c22f-48ce-a2b3-6209225681c8.jpg)


## Data Overview

Once the file is uploaded, you can have an overview of the file regarding its shape, null values and other features. <br />

![file_uploaded](https://user-images.githubusercontent.com/72215169/171049028-a1b4eaa3-0bd7-4117-adfc-9d45d692f763.jpg)


## Data Pre-processing

The app offers various data pre-processing services ranging from dropping rows columns, filling in missing values to label encoding columns.


![features](https://user-images.githubusercontent.com/72215169/171049041-cee8be8b-0d42-43ed-ab68-d56520fdbb3e.jpg)


![missing](https://user-images.githubusercontent.com/72215169/171049048-920c071a-11c0-487e-b6f5-a5a7b8d6b3ce.jpg)





![plots](https://user-images.githubusercontent.com/72215169/171049057-f201885b-5664-4ba6-9a9d-268c55c91bbc.jpg)




![outliers](https://user-images.githubusercontent.com/72215169/171049067-3d34dd59-8a6f-4535-8bbf-41185c06aba1.jpg)




![drop](https://user-images.githubusercontent.com/72215169/171049087-342a35a7-b995-40fe-9121-5e835472ce80.jpg)




![label](https://user-images.githubusercontent.com/72215169/171049091-d9b012d8-6bb4-4fef-a7ab-515e6c172b86.jpg)


## Machine Learning Model

### Regression Models

![regression](https://user-images.githubusercontent.com/72215169/171049099-fddbd181-e224-4583-9ec4-298f14359ec2.jpg)

There are 5 regression models that you can play with on their parameters.
Ploynomial Linear Regression <br />
Multiple Linear Regression <br />
Decision Tree Regression <br />
Random Forest Regression <br />
Support Vector Regression <br />

![types](https://user-images.githubusercontent.com/72215169/171049119-cd1a0075-4faf-48e8-aa63-1e0c99d7a41d.jpg)

### Classification Models


![classification](https://user-images.githubusercontent.com/72215169/171049126-a55ca6d2-adec-447b-a4cc-c0ed06664ebb.jpg)

There are 7 classification models that you can play with on their parameters.
Logistic Regression <br />
Decision Tree Classifier <br />
Random Forest Classifier <br />
Naive Bayes <br />
KNN <br />
Linear Discriminant Analysis <br />
Kernel SVM <br />

![classify_all](https://user-images.githubusercontent.com/72215169/171049142-cc44ef00-069e-4899-b101-7620bd2f6bca.jpg)





