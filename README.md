This project tends to create tools that would be valuable for data analysis and calssification. 

First features with low correlation are selected.

Function that plots numerical colums provide information about data distribution and outliers.
Middle: unique values without outliers.
Higher: higher outliers.
Lower: lower outliers.
Red line represent positive - in our case the onse that did not get loans.
Green negative.
Proportion here is positive/negative, depending on data distribution.

Downsampling is done using accending of valuable columns and modulus.

SHAP values provide valuable insights into feature importance: ttps://arxiv.org/pdf/1802.03888.pdf



In Machine learning part multiple classifier are created and tested to get best performance.
First base classifier is tested with application data.
Aggregation of historical information is performed.
As LGB Calssifier handles outliers and NaN values, only anomalies where removed during feature engineering.
Final roc auc score in Kaggle late sumbision is 0.77.


Classifier performance is provided in classification reports.

Precision proportion of positive predictions that were actually correct.
Recall proportion of actual positive cases that were correctly identified.
The f1-score is the harmonic mean between precision & recall
The support is the total number of true instances for a particular class.

In this scenario one way would be to divide data into high amount loans and low amountloans.

Then focus on better recall for high amount loans as risk of false negative would give bigger impact.

