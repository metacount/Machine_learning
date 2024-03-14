    Project tends to create valuable tools for data analysis and classification. 
    Used datset: https://www.kaggle.com/c/home-credit-default-risk/

    EDA notebook
    
    First features with low correlation are selected.
    Plot numerical distribution and outliers.
    Downsampling is done using accendance of valuable columns and modulus.
    Plot Shap Values, gradient boosting tree and categorical features.
    For multivariant analysis check Colab link in EDA notebook.

     
    Machine Learning notebook
    
    Multiple classifier are created and tested to get best performance.
    Classifiers are mesuare by Area Under The Curve score.
    First base classifier is tested with application data, which provided 0.67 score.
    Aggregation of historical information is performed.
    As LGB Calssifier handles outliers and NaN values, only anomalies removed.
    Final score in Kaggle late sumbision is 0.77.
    Classifier performance is provided in classification reports.


    EDA and classification impact on buisness
    
    it is assumed that current classification is source of truth.
    aditional attention should be given to outliers in prodcution environment.
    divide data into high amount and low amount loans.
    target for better recall for high amount loans. 
    display of shap values for individual predictions.
    predictions can be divided into: positive, negative and uncertain.
    classifier should be trained on as big data set as possible.

    
    Security, monitoring and scalability should be consider together with customer.


    Dependencies:
    this project was done using Google Colab, one library to be installed is shap.
