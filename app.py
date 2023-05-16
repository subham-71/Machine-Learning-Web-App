from audioop import mul
from cmath import nan
from datetime import date
import streamlit as st
from helper import data, describe, outliers, drop_items, download_data, filter_data, num_filter_data, rename_columns, clear_image_cache, handling_missing_values,label_encode
from ml import d_tree_regression,poly_regression,multi_regression, random_forest_regression, svr, classification, kernel_svm,logistic , random_forest, naive_bayes, KNN, d_tree_classification,linear_disc
import numpy as np
import pandas as pd

st.set_page_config(
     page_title="Data Pre-processing and Model Selection Web App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     
)

st.sidebar.title("PreproModel")

file_format_type = ["csv", "txt", "xls", "xlsx", "ods", "odt"]
functions = ["Overview",  "Drop Columns", "Drop Categorical Rows", "Drop Rows in Range", "Rename Columns", "Display Plot","Outliers", "Handle Missing Data","Label Encode" ]
excel_type =["vnd.ms-excel","vnd.openxmlformats-officedocument.spreadsheetml.sheet", "vnd.oasis.opendocument.spreadsheet", "vnd.oasis.opendocument.text"]
model_types =['Regression' , 'Classification']
uploaded_file = st.sidebar.file_uploader("Upload Your file", type=file_format_type)

if uploaded_file is not None:

    file_type = uploaded_file.type.split("/")[1]
    
    if file_type == "plain":
        seperator = st.sidebar.text_input("Please Enter what seperates your data: ", max_chars=5) 
        data = data(uploaded_file, file_type,seperator)

    elif file_type in excel_type:
        data = data(uploaded_file, file_type)

    else:
        data = data(uploaded_file, file_type)
    
    describe, shape, columns, num_category, str_category, null_values, dtypes, unique, str_category, column_with_null_values = describe(data)

    st.sidebar.write("### Data Preprocessing")
    multi_function_selector = st.sidebar.multiselect("Select the type of function to perform on this dataset ",functions, default=["Overview"])


    if "Overview" in multi_function_selector:
        st.subheader("Dataset Preview")
        st.dataframe(data)
        st.subheader("Dataset Description")
        st.write(describe)

        st.text(" ")
        st.text(" ")
        st.text(" ")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.text("Basic Information")
            st.write("Dataset Name")
            st.text(uploaded_file.name)

            st.write("Dataset Shape")
            st.write(shape)
            
        with col2:
            st.text("Dataset Columns")
            st.write(columns)
        
        with col3:
            st.text("Numeric Columns")
            st.dataframe(num_category)
        
        with col4:
            st.text("String Columns")
            st.dataframe(str_category)
            

        col5, col6, col7= st.columns(3)

        with col5:
            st.write("Null Values count")
            st.dataframe(null_values)
        with col6:
            st.text("Data-Types")
            st.dataframe(dtypes)
        
        with col7:
            st.text("Unique Values count")
            st.write(unique)
    
# to detect outliers

    if "Outliers" in multi_function_selector:
        st.write("## Outliers")
        outliers_selection = st.multiselect("Select Columns to see their outliers", num_category)
        outliers = outliers(data, outliers_selection)
        
        for i in range(len(outliers)):
            st.image(outliers[i])

# to drop columns 

    if "Drop Columns" in multi_function_selector:
        st.write("## Drop columns")
        multiselected_drop = st.multiselect("Select name of columns to drop ", data.columns)
        
        droped = drop_items(data, multiselected_drop)
        st.write(droped)
        
        drop_export = download_data(droped, label="Droped(edited)")
# to drop columns 

    if "Label Encode" in multi_function_selector:
        st.write("## Label Encoding")
        column = st.selectbox("Select name of columns encode ", str_category)
        
        df = label_encode(data, column)
        st.write(df)
        
        drop_export = download_data(df, label="Lable_Encoded")

# to drop particular values in a column

    if "Drop Categorical Rows" in multi_function_selector:
        st.write("Drop rows")

        column = st.selectbox("Select column: ", options=data.columns)
        drop_values = st.multiselect("Enter Name or Select the value which you don't want in your {} column(You can choose multiple values): ".format(column), data[column].unique())
        
        df_new = filter_data(data, column, drop_values)
        st.write(df_new)
        
        df_download = download_data(df_new, label="filtered")

# Drop numeric values (useful for outliers)

    if "Drop Rows in Range" in multi_function_selector:
        st.write("Drop numeric rows in a range")

        option = st.radio(
        "Which kind of Filteration you want",
        ('Delete data inside the range', 'Delete data outside the range'))

        num_column = st.selectbox("Select column: ", options=num_category)
        selection_range = data[num_column].unique()

        for i in range(0, len(selection_range)) :
            selection_range[i] = selection_range[i]
        selection_range.sort()

        selection_range = [x for x in selection_range if np.isnan(x) == False]

        start_value, end_value = st.select_slider(
        'Select range of numbers you want to edit or keep',
        options=selection_range,
        value=(min(selection_range), max(selection_range)))
        
        if option == "Delete data inside the range":
            st.write('We will be removing all the values between ', int(start_value), 'and', int(end_value))
            num_filtered_data = num_filter_data(data, start_value, end_value, num_column, param=option)
        else:
            st.write('We will be Keeping all the values between', int(start_value), 'and', int(end_value))
            num_filtered_data = num_filter_data(data, start_value, end_value, num_column, param=option)

        st.write(num_filtered_data)
        num_filtered_export = download_data(num_filtered_data, label="num_filtered")


# Renaming Columns

    if "Rename Columns" in multi_function_selector:
        st.write("## Rename columns")
        if 'rename_dict' not in st.session_state:
            st.session_state.rename_dict = {}

        rename_dict = {}
        rename_column_selector = st.selectbox("Select column to rename: ", options=data.columns)
        rename_text_data = st.text_input("Enter New Name for the {} column".format(rename_column_selector), max_chars=50)


        if st.button("Draft Changes", help="when you want to rename multiple columns/single column  so first you have to click Save Draft button this updates the data and then press Rename Columns Button."):
            st.session_state.rename_dict[rename_column_selector] = rename_text_data
        st.code(st.session_state.rename_dict)

        if st.button("Apply Changes", help="Takes your data and rename the column as your wish."):
            rename_column = rename_columns(data, st.session_state.rename_dict)
            st.write(rename_column)
            export_rename_column = download_data(rename_column, label="rename_column")
            st.session_state.rename_dict = {}

# Visualization using Plots
 
    if "Display Plot" in multi_function_selector:
        st.write("## Display Plot")
        st.write("Plots graphs with one data type as string")
        multi_bar_plotting = st.multiselect("Select Column to Plot: ", str_category)
        
        for i in range(len(multi_bar_plotting)):
            column = multi_bar_plotting[i]
            st.markdown("#### Bar Plot for {} column".format(column))
            bar_plot = data[column].value_counts().reset_index().sort_values(by=column, ascending=False)
            st.bar_chart(bar_plot)

# Handling missing values    

    if "Handle Missing Data" in multi_function_selector:
        st.write("## Handle Missing Values")
        handling_missing_value_option = st.radio("Select method", ("Drop Null Values", "Filling in Missing Values"))

        if handling_missing_value_option == "Drop Null Values":

            drop_null_values_option = st.radio("Choose your option as suted: ", ("Drop all null value rows", "Only Drop Rows that contanines all null values"))
            droped_null_value = handling_missing_values(data, drop_null_values_option)
            st.write(droped_null_value)
            export_rename_column = download_data(droped_null_value, label="dropped_data_column")
        
        elif handling_missing_value_option == "Filling in Missing Values":
            
            if 'missing_dict' not in st.session_state:
                st.session_state.missing_dict = {}
           
            st.write("### Statistics for numerical columns ")
            st.write("Helps to fill in the null values ")
            st.write(data.describe())

            fillna_column_selector = st.selectbox("Select column Name you want to fill the NaN Values: ", options=column_with_null_values)
            fillna_text_data = st.text_input("Enter the New Value for the {} Column NaN Value".format(fillna_column_selector), max_chars=50)

            if st.button("Draft Changes", help="when you want to fill multiple columns/single column null values so first you have to click Save Draft button this updates the data and then press Rename Columns Button."):     
                
                if fillna_column_selector in num_category:
                    try:
                        st.session_state.missing_dict[fillna_column_selector] = float(fillna_text_data)
                    except:
                        st.session_state.missing_dict[fillna_column_selector] = int(fillna_text_data)
                else:
                    st.session_state.missing_dict[fillna_column_selector] = fillna_text_data

            st.code(st.session_state.missing_dict)

            if st.button("Apply Changes", help="Takes your data and Fill NaN Values for columns as your wish."):

                fillna_column = handling_missing_values(data,handling_missing_value_option, st.session_state.missing_dict)
                st.write(fillna_column)
                export_rename_column = download_data(fillna_column, label="fillna_column")
                st.session_state.missing_dict = {}
# ML Model 
# if st.sidebar.button("Build Machine Learning Model"):
    st.sidebar.write("### Machine Learning Model")
    st.sidebar.write("Make sure to place your target column as the last column in your csv file")
    build_model = st.sidebar.selectbox("Select method",model_types)
    st.write("# Machine Learning Model ")
    st.write("Make sure to place your target column as the last column in your csv file")

    regression_models = ['Polynomial Linear Regression','Multiple Linear Regression','Decision Tree Regression','Random Forest Regression','Support Vector Regression' ]

    if build_model=="Regression":
             regression_model = st.selectbox("Select Regression Model ", options=regression_models)
             if(regression_model=='Decision Tree Regression'):
                 selection_range_random_state = [x for x in range(100)]
                 select_test_size = [float(x/100) for x in range(20,100)]
                #  print(select_test_size)
                 test_size = st.select_slider('Select test data size', options=select_test_size)
                 random_val = st.select_slider('Select random state value', options=selection_range_random_state)
                 acc_score = d_tree_regression(data,random_val,test_size)
                 st.write("# Accuracy score : ",acc_score)

             if(regression_model=='Polynomial Linear Regression'):
                 select_test_size = [float(x/100) for x in range(20,100)]
                 degree_rang = [x for x in range(1,15)]
                 test_size = st.select_slider('Select test data size', options=select_test_size)
                 degree = st.select_slider('Select polynomial degree', options=degree_rang)
                 acc_score = poly_regression(data,test_size,degree)
                 st.write("# Accuracy score : ",acc_score)

             if(regression_model=='Multiple Linear Regression'):
                 select_test_size = [float(x/100) for x in range(20,100)]
                 test_size = st.select_slider('Select test data size', options=select_test_size)
                 acc_score = multi_regression(data,test_size)
                 st.write("# Accuracy score : ",acc_score)

             if(regression_model=='Random Forest Regression'):
                 selection_range_random_state = [x for x in range(100)]
                 select_test_size = [float(x/100) for x in range(20,100)]
                 estimators_range = [x for x in range(5,100)]

                 test_size = st.select_slider('Select test data size', options=select_test_size)
                 random_val = st.select_slider('Select random state value', options=selection_range_random_state)
                 estimators = st.select_slider('Select n_estimators', options=estimators_range)

                 acc_score = random_forest_regression(data,random_val,test_size,estimators)
                 st.write("# Accuracy score : ",acc_score)

             if(regression_model=='Support Vector Regression'):
                 selection_range_random_state = [x for x in range(100)]
                 select_test_size = [float(x/100) for x in range(20,100)]
                 e_params_range = [float(x/10) for x in range(1,500)]
                 c_params_range = [float(x/10) for x in range(10,500)]
                 kernels = ['rbf','linear', 'poly',  'sigmoid', 'precomputed']

                 test_size = st.select_slider('Select test data size', options=select_test_size)
                 e_params = st.select_slider('Select epsilon value', options=e_params_range)
                 c_params = st.select_slider('Select regularization parameter (c)', options=c_params_range)
                 kernel = st.selectbox("Select kernel", options=kernels)

                 acc_score = svr(data,kernel,test_size,c_params,e_params)
                 st.write("# Accuracy score : ",acc_score)

    classification_models= ['Logistic','Random Forest','Naive Bayes','Kernel SVM','KNN', 'Decision Tree','Linear Discriminant Analysis' ,'Compare All']



            
    if build_model=="Classification":

             classification_model = st.selectbox("Select classification model ", options=classification_models)

             if(classification_model=='Compare All'):
                select_test_size = [float(x/100) for x in range(20,100)]
                test_size = st.select_slider('Select test data size', options=select_test_size)                 
                df_ans=classification(data,test_size)
                st.dataframe(df_ans)

             if(classification_model=='Kernel SVM'):

                 selection_range_random_state = [x for x in range(100)]
                 select_test_size = [float(x/100) for x in range(20,100)]
                 kernels = ['rbf','linear', 'poly', 'sigmoid']

                 test_size = st.select_slider('Select test data size', options=select_test_size)
                 random_val = st.select_slider('Select random state value', options=selection_range_random_state)
                 kernel = st.selectbox("Select kernel", options=kernels)

                 acc_score,precision,recall ,f1 = kernel_svm(data,test_size,kernel,random_val)
                 st.write("#### Accuracy score : ",acc_score)
                 st.write("#### Precision : ",precision)
                 st.write("#### Recall  : ",recall)
                 st.write("#### F1  : ",f1)

             if(classification_model=='Logistic'):

                 selection_range_random_state = [x for x in range(100)]
                 select_test_size = [float(x/100) for x in range(20,100)]
                
                 test_size = st.select_slider('Select test data size', options=select_test_size)
                 random_val = st.select_slider('Select random state value', options=selection_range_random_state)
                 
                 acc_score,precision,recall ,f1 = logistic(data,test_size,random_val)
                 st.write("#### Accuracy score : ",acc_score)
                 st.write("#### Precision : ",precision)
                 st.write("#### Recall  : ",recall)
                 st.write("#### F1  : ",f1)

             if(classification_model=='Random Forest'):

                 selection_range_random_state = [x for x in range(100)]
                 select_test_size = [float(x/100) for x in range(20,100)]
                 estimators_range = [x for x in range(5,100)]
                 criteria_range = ['entropy','gini','log_loss']

                 test_size = st.select_slider('Select test data size', options=select_test_size)
                 random_val = st.select_slider('Select random state value', options=selection_range_random_state)
                 estimators = st.select_slider('Select n_estimators', options=estimators_range)
                 criteria = st.selectbox("Select criteria", options=criteria_range)    

                 acc_score,precision,recall ,f1 = random_forest(data,test_size,random_val,estimators,criteria)
                 st.write("#### Accuracy score : ",acc_score)
                 st.write("#### Precision : ",precision)
                 st.write("#### Recall  : ",recall)
                 st.write("#### F1  : ",f1)

             if(classification_model=='Naive Bayes'):

                 
                 select_test_size = [float(x/100) for x in range(20,100)]
                 test_size = st.select_slider('Select test data size', options=select_test_size)
             
                 acc_score,precision,recall ,f1 = naive_bayes(data,test_size)
                 st.write("#### Accuracy score : ",acc_score)
                 st.write("#### Precision : ",precision)
                 st.write("#### Recall  : ",recall)
                 st.write("#### F1  : ",f1)
            
             if(classification_model=='KNN'):

                 select_test_size = [float(x/100) for x in range(20,100)]
                 neighbors_range = [x for x in range(3,100)]
                 
                 test_size = st.select_slider('Select test data size', options=select_test_size)
                 neighbors = st.select_slider('Select number of neighbors', options=neighbors_range)

                 acc_score,precision,recall ,f1 = KNN(data,test_size,neighbors)
                 st.write("#### Accuracy score : ",acc_score)
                 st.write("#### Precision : ",precision)
                 st.write("#### Recall  : ",recall)
                 st.write("#### F1  : ",f1)

             if(classification_model=='Decision Tree'):

                 selection_range_random_state = [x for x in range(100)]
                 select_test_size = [float(x/100) for x in range(20,100)]
                 criteria_range = ['entropy','gini','log_loss']

                 test_size = st.select_slider('Select test data size', options=select_test_size)
                 random_val = st.select_slider('Select random state value', options=selection_range_random_state)
                 criteria = st.selectbox("Select criteria", options=criteria_range)    

                 acc_score,precision,recall ,f1 = d_tree_classification(data,test_size,random_val,criteria)
                 st.write("#### Accuracy score : ",acc_score)
                 st.write("#### Precision : ",precision)
                 st.write("#### Recall  : ",recall)
                 st.write("#### F1  : ",f1)

             if(classification_model=='Linear Discriminant Analysis'):

                 select_test_size = [float(x/100) for x in range(20,100)]
                 solver_range = ["svd","lsqr","eigen"]

                 test_size = st.select_slider('Select test data size', options=select_test_size)
                 solver = st.selectbox("Select solver", options=solver_range)    
                 
                 acc_score,precision,recall ,f1 = linear_disc(data,test_size,solver)
                 st.write("#### Accuracy score : ",acc_score)
                 st.write("#### Precision : ",precision)
                 st.write("#### Recall  : ",recall)
                 st.write("#### F1  : ",f1)

    st.sidebar.info("After using this app, click Clear Cache so that your all data is removed from the folder.")
    if st.sidebar.button("Clear Cache"):
        clear_image_cache()

else:
    with open('samples/sample.zip', 'rb') as f:
        st.sidebar.download_button(
                label="Download Sample Data and Use It",
                data=f,
                file_name='smaple_data.zip',
                help = "Download some sample data and use it to explore this web app."
            )
