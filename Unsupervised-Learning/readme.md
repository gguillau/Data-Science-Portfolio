# Unsupervised Learning

 The telecom operator Interconnect would like to be able to forecast their churn of clients. If it's discovered that a user is planning to leave, they will be offered promotional codes and special plan options. Interconnect's marketing team has collected some of their clientele's personal data, including information about their plans and contracts.

## Interconnect's Services
 Interconnect mainly provides two types of services:

* Landline communication. The telephone can be connected to several lines simultaneously.
* Internet. The network can be set up via a telephone line (DSL, digital subscriber line) or through a fiber optic cable.

 Some other services the company provides include:

 * Internet security: antivirus software (DeviceProtection) and a malicious website blocker (OnlineSecurity)
 * A dedicated technical support line (TechSupport)
 * Cloud file storage and data backup (OnlineBackup)
 * TV streaming (StreamingTV) and a movie directory (StreamingMovies)
 * The clients can choose either a monthly payment or sign a 1- or 2-year contract. They can use various payment methods and receive an electronic invoice after a transaction.
 ## Objective
 * Forecast the churn of clients
 * Collect the necessary information to assist the marketing department in figuring out different ways of retaining clients.
 * Compare the monthly payment distribution (MonthlyCharges) of all active clients with the clients who have left. 
 * Calculate the statistics for each group: The average, minimum and maximum values, the median, and the values of the 25% and 75% percentiles. 
 * Build distribution histograms based on your findings.
 * Compare the behavior of the clients from the two groups below. For each group, build any two graphs which display: The share of telephone users, The share of Internet users

 ## Data Description
 The data consists of files obtained from different sources. In each file, the column customerID contains a unique code assigned to each client. The contract information is valid as of February 1, 2020.
 * contract.csv — contract information
 * personal.csv — the client's personal data
 * internet.csv — information about Internet services
 * phone.csv — information about telephone services

  - Target feature: the 'EndDate' column equals 'No'.
  - Primary metric: AUC-ROC.
  - Additional metric: Accuracy.

##  Libraries Used
 * Pandas
 * Matplotlib.pyplot
 * scipy.stats
 * numpy
 * collections
 * sklearn
 * scipy
 * scikitplot
 * datetime
 * xgboost
 * lightgbm
 * catboost

##  Models Evaluated
 * DecisionTreeClassifier
 * RandomForestClassifier
 * AdaBoostClassifier
 * XGBClassifier
 * CatBoostClassifier
 * LGBMClassifier
 

## Background
This project is part of the Data Scientist training program from Practicum by Yandex. More info in link below:

https://practicum.yandex.com/data-scientist

 
