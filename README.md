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
Classifiers are mesuare by Area Under The Curve for Receiver Operating Characteristics(auc) score.
First base classifier is tested with application data, which provided 0,67 score.
Aggregation of historical information is performed.
As LGB Calssifier handles outliers and NaN values, only anomalies where removed during feature engineering.
Final score in Kaggle late sumbision is 0.77.


Classifier performance is provided in classification reports.

Precision proportion of positive predictions that were actually correct.
Recall proportion of actual positive cases that were correctly identified.
The f1-score is the harmonic mean between precision & recall
The support is the total number of true instances for a particular class.

In this scenario one way would be to divide data into high amount and low amount loans.
Then target for better recall for high amount loans as risk of false negative would give bigger impact.

Regarding eda and classification impact on buisness:

it is assumed that current classification is source of truth.
aditional attention should be given to outliers.
display of shap values for individual prediction, so that reviewer can see which client featues are most impactful.
predictions can be divided into three categories, positive, negative and uncertain.
application infrsatructure could be either in cloud or on premise in customer datacenter.
classifier should be trained on as big data set as possible.
when new application is provided prediction is done using appcation and historical aggregated client data.

Security, monitoring and scalability should be consider together with customer.
