# Machine-Learning-Web-App
A web app to simplify the data pre processing steps and hyper parameter tuning process of machine learning models. It helps in selecting the best model with optimal parameters.
The web app has services for both data pre-processing steps and machine learning model hypertuning process.

## File Upload
First a file is uploaded to the web-app. The file for pre-processing can be raw. But for building a machine learning model and selecting the best model with optiimal parameters, a pre-processed file needs to be uploaded with the target feature at its last column. It can be either a regression or a classification task.

File upload format : cdv,txt,xls,xlsx,ods,odt<br />
I have also given link to download sample datasets so that explore the app.<br />
Sample Datasets :

sample_file_for_preprocessing    : You can use this file for pre-processing tasks <br />
regression_preprocessed_file     : You can use this file to build regression models <br />
classification_preprocessed_file : You can use this file to build classification models <br />

![Screenshot 2022-05-30 232816](https://user-images.githubusercontent.com/72215169/171046080-9e68f1ec-0f24-4a27-ac22-3d1fb849f61c.jpg)


## Data Overview

Once the file is uploaded, you can have an overview of the file regarding its shape, null values and other features. <br />

![data_uploaded](https://user-images.githubusercontent.com/72215169/171046160-246ed569-e2cd-4a5d-9d5b-de2bcd312813.jpg)

## Data Pre-processing

The app offers various data pre-processing services ranging from dropping rows columns, filling in missing values to label encoding columns.

![features](https://user-images.githubusercontent.com/72215169/171046321-fae1d6ff-a6d6-48ce-836a-30c61b2ece29.jpg)

### Handling Missing Values

![missing_values](https://user-images.githubusercontent.com/72215169/171046520-b5c862e5-dfb0-4156-b66c-066debc35279.jpg)

### Visualizing through plots

![plots](https://user-images.githubusercontent.com/72215169/171046550-61bec6c3-b23f-47c6-81cf-7f72800e577b.jpg)

### Viewing Outliers

![checking outliers](https://user-images.githubusercontent.com/72215169/171046641-87cc147e-0001-405c-bd62-44633e1f1602.jpg)


### Drop Rows and Columns

![drop rows and columns](https://user-images.githubusercontent.com/72215169/171046476-846601e8-900f-4f1d-abce-c680183dc379.jpg)


### Label Encoding columns

![label_encoding](https://user-images.githubusercontent.com/72215169/171046696-ba8d364f-878a-4de2-ade1-e3b64234c6a5.jpg)


## Machine Learning Model

### Regression Models

![regression](https://user-images.githubusercontent.com/72215169/171046779-051d4f40-9ac5-4783-bf95-8ce834429028.jpg)

There are 5 regression models that you can play with on their parameters.
Ploynomial Linear Regression <br />
Multiple Linear Regression <br />
Decision Tree Regression <br />
Random Forest Regression <br />
Support Vector Regression <br />

![types of regression models](https://user-images.githubusercontent.com/72215169/171046778-a65ad7ff-1470-4ca2-a947-10281a467a6d.jpg)

### Classification Models

![classification_file_uploaded](https://user-images.githubusercontent.com/72215169/171047111-8dfc7849-40ed-43a0-814f-c5840826358b.jpg)

There are 7 classification models that you can play with on their parameters.
Logistic Regression <br />
Decision Tree Classifier <br />
Random Forest Classifier <br />
Naive Bayes <br />
KNN <br />
Linear Discriminant Analysis <br />
Kernel SVM <br />


![classification_model_metrics](https://user-images.githubusercontent.com/72215169/171047123-ff4b2010-874a-4f3d-9026-8cd93f086e03.jpg)



