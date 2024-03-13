    Project tends to create valuable tools for data analysis and calssification. 

    EDA notebook
    
    First features with low correlation are selected.
    Plot numerical colums provide information about data distribution and outliers.
    Middle: unique values without outliers.
    Higher: higher outliers.
    Lower: lower outliers.
    Red line represent positive - in our case the onse that did not get loans.
    Green negative.
    Proportion here is positive/negative, depending on data distribution.
    Downsampling is done using accendance of valuable columns and modulus.
    SHAP values: https://arxiv.org/pdf/1802.03888.pdf

    
    Machine Learning notebook
    
    multiple classifier are created and tested to get best performance.
    Classifiers are mesuare by Area Under The Curve score.
    First base classifier is tested with application data, which provided 0.67 score.
    Aggregation of historical information is performed.
    As LGB Calssifier handles outliers and NaN values, only anomalies removed.
    Final score in Kaggle late sumbision is 0.77.
    
    
    Classifier performance is provided in classification reports.
    
    Precision proportion of positive predictions that were actually correct.
    Recall proportion of actual positive cases that were correctly identified.
    The f1-score is the harmonic mean between precision & recall.
    The support is the total number of true instances for a particular class.


    Regarding eda and classification impact on buisness
    
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
