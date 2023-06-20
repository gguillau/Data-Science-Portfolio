# <p align=center> Data Science & Analytics Project Portfolio </p>
---

Repository containing portfolio of ongoing and completed data science projects completed by me for the [Practicum Data Science bootcamp](https://practicum.yandex.com/data-scientist) and academic, self learning, and hobby purposes.

Following are the highlights of the projects:
- **Languages/Software**: *Python, PostgreSQL, HTML, CSS, SPSS Statistics, Microsoft Excel/Access, Qualtrics Surveys, Google Analytics, Adobe Analytics*

- __Tools__: *pandas, NumPy, seaborn, matplotlib, plotly, sciPy, scikit-learn, TensorFlow, Keras, geopandas, folium, langid, Beautiful Soup,  Selenium, transformers, NLP (NLTK, spaCy, BERT), librosa, spotipy, PySpark, electronJs, shapely, SSH, SFTP, Unix/Linux*
  

* __Machine Learning Models Evaluated__:
    * __Classification Models__: *DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, AdaBoostClassifier, XGBClassifier, CatBoostClassifier, LGBMClassifier, k-NNClassifier*
    * __Regression Models__: *LinearRegressor, DecisionTreeRegressor, RandomForestRegressor, LogisticRegression, XGBRegressor, LGBMRegressor, CatBoostRegressor*

- **Psychological Assessment Scales and Measures:**
   * *Multigroup Ethnic Identity Measure (MEIM)*
   * *State-Trait Anxiety Inventory (STAI)*

|    Hands-on Experience |         Project       | Technical skills       | 
|   -------------------- |      -------------   |--------------- | 
| Unsupervised Learning  | [Client Churn Forecast w/ Machine Learning](https://github.com/giova22i/Data-Science-Portfolio/tree/main/Unsupervised-Learning) | Machine learning algorithms, XGBoost, CatBoost, LightGBM | 
| Computer Vision (CV)  | [Computer Vision Age Detection w/ deep learning](https://github.com/giova22i/Data-Science-Portfolio/tree/main/Computer-Vision) | Tensorflow |
| Natural Language Processing (NLP)  |  [IMDB Movie Sentiment Analysis using NLP](https://github.com/giova22i/Data-Science-Portfolio/tree/main/ML-for-Texts)   | SGDClassifier, Naïve bayes, LightGBM, spaCy, TF-IDF, BERT |    
|   Time Series Analysis  |  [Time Series Forecast](https://github.com/giova22i/Data-Science-Portfolio/tree/main/Time-Series-Project)          | Time Series Analysis, CatBoost, LightGBM, XGBoost |
|   Machine Learning in Business |  [Gold Recovery Regression Model](https://github.com/giova22i/Gold-Recovery-ML)          | Python, Scikit-learn, LinearRegression |   
  |   Numerical Methods with ML  |[Vehicle Market Value Prediction w/ gradient boosting](https://github.com/giova22i/Data-Science-Portfolio/tree/main/Numerical-Methods-Project)           | Numerical Methods, CatBoost, LightGBM, XGBoost |
|   Linear Algebra with Machine Learning  | [Insurance Benefits Predictive Model](https://github.com/giova22i/Data-Science-Portfolio/tree/main/Linear-Algebra-Project)   | Scikit-learn, Linear Algebra, k-Nearest Neighbors |
|   Machine Learning - Classification | [Telecom Plan Classification Model](https://github.com/giova22i/introML)           | Python, Scikit-learn, Pandas |
|   Supervised Learning - Prediction  |  [Bank Customer Churn Prediction](https://github.com/giova22i/supervisedlearning)          | Scikit-learn, XGBoost, GridSearchCV, AdaBoost |    
|   Machine Learning in Business | [Oil Well Regression Model](https://github.com/giova22i/MLB)          | Python, Scikit-learn, Bootstrapping, LinearRegression |
|   Webscraping and Data Storage      | [Data Collection, Webscraping and Storage](https://github.com/giova22i/SQL-Project)           | PostgreSQL, Python, BeautifulSoup, Seaborn, Matplotlib, ETL (extract, transform and load) | 
|   Data Visualization and Storytelling with Data       | [Video Game Market Analysis](https://github.com/giova22i/Video-Game-Sales-Analysis)          | Python, Pandas, Squarify, Seaborn, Matplotlib |   
|   Data Preprocessing   |            [Credit Score Analysis](https://github.com/giova22i/Credit-Report-Analysis)           | Python,  NLTK, WordNetLemmatizer, SnowballStemmer, Seaborn, Matplotlib | 
|   Exploratory Data Analysis (EDA)           | [Vehicle Market Analysis](https://github.com/giova22i/Practicum-EDA)         | Pandas, Matplotlib | 
|   Statistical Data Analysis (SDA)    | [Telecom Customer Data Analysis](https://github.com/giova22i/Prepaid-Plan-Analysis)           | Python, pandas, Numpy, SciPy, Seaborn, Matplotlib 





##  <p align=center> Work Experience: </p>
### [Tweet Geolocation Prediction - Yachay.ai](https://github.com/giova22i/Yachay.ai-Tweet-Geolocation-Prediction)
Contributed research to company’s infrastructure, with the goal of training a deep learning model using BERT to predict user geolocation from individual tweets.

__Highlights__:

* Identified and gathered relevant data from various sources.
* Performed exploratory data analysis to gain insights into the data.Implemented Hugging Face NLP pipelines to extract text features: sentiment, topic, language
* Preprocess the text data (e.g., tokenization, removing stop words) using BERT  for model training
* Evaluated model performance using appropriate MSE and haversine loss
* Median and mean differences between predicted and actual distances were 1,334 km and 1,881 km, respectively, demonstrating the model's accuracy.
* Acknowledged the workflow's potential to provide valuable geolocation prediction capabilities, with the possibility of scaling and integrating it into the existing infrastructure for real-time application

![alt text](https://github.com/giova22i/Yachay.ai-Tweet-Geolocation-Prediction/blob/main/charts/mintplot.png)

*Tools:  Python, pandas, seaborn,  scikit-learn, langid, geopandas, tensorflow, BERT*

** *
### [Prediciting Song Valence - Cuetessa,inc](https://github.com/giova22i/Cuetessa-Song-Valence-Prediction)

Tasked with developing a Python-based regression model to predict the valence of pop songs for playlist curation and other applications. An automatic method of classifying the valence of pop songs is useful for playlist curation and other applications.


__Highlights__:
* Data collection and extraction using Spotify’s Web API 
* Audio feature extraction and analysis from mp3 files using Librosa (python package for music analysis)
* Regression analysis to predict valence using both song lyrics (NLP) and audio features as input
* Regression analysis to predict valence using song lyrics (NLP) and audio features as input
* Implemented various approaches to train and validate models to forecast valence scores of songs
* Conducted model training, validation, and hyperparameter optimization using RandomSearchCV on four regression models: Random Forest, K-Nearest Neighbors, XGBoost, and Linear Regression.
* Selected Support Vector Regression model trained on 191 normalized audio features, achieving an RMSE score of 0.16 on testing data, meeting the company's desired model performance standards
* Learned to use AWS for Machine learning, including driver management
 
   
<p align="center">
  <img src="https://github.com/giova22i/Cuetessa-Song-Valence-Prediction/blob/main/charts/resultsvalence.png" />
</p>

*Tools: pandas, numPy, matplotlib, seaborn, spotipy, transformers, sklearn, Librosa*

** *


 ### [Grant Automated Web Scraper - DataReadyDFW](https://github.com/giova22i/DataReady-Internship)
 
 Researching grant prospects can be time-consuming and overwhelming. Develop an automation system for a nonprofit organization (DataReady DFW) to find available grant opportunities and fill out applications with little or no human intervention. 


__Highlights__:
* Database management; collected, analyzed, and interpreted raw public grant data 
* Developed and implemented scripts to autofill constant values 
* Collection, reporting, and analysis of website data
* Filtered/trained script to select further grant opportunities using Natural Language Processing
 
*Tools: TensorFlow, Beautiful Soup, Selenium, pandas, ntlk, Google Analytics*

##  <p align=center> Technical Projects: </p>

* ### **Machine Learning (Python)**
    * [Computer Vision Age Detection](https://github.com/giova22i/Data-Science-Portfolio/tree/main/Computer-Vision): Built a regression model for computer vision, in order to predict the approximate age of a customer from a Supermarket checkout  photograph.
    * [IMDB Movie Sentiment Analysis using NLP](https://github.com/giova22i/Data-Science-Portfolio/tree/main/ML-for-Texts): Build a machine learning model to automatically detect negative reviews for a system used to filter and categorize movie reviews.
    * [Time Series Forecast](https://github.com/giova22i/Data-Science-Portfolio/tree/main/Time-Series-Project): Use historical data on taxi orders at airports to create a model that predicts the number of taxi orders for the next hour.
    * [Vehicle Market Value Prediction](https://github.com/giova22i/Data-Science-Portfolio/tree/main/Numerical-Methods-Project): Determine the market value of a used car using historical car data. Identify the quality and speed of prediction for various models.
    * [Insurance Benefits Predictive Model](https://github.com/giova22i/Data-Science-Portfolio/tree/main/Linear-Algebra-Project): Linear regression for insurance benefits prediction
    * [Gold Recovery Regression Model](https://github.com/giova22i/Gold-Recovery-ML): Prepare a prototype of a machine learning model for Zyfra  to predict the amount of gold recovered from the gold ore for the purpose of optimizing production and eliminating unprofitable parameters. The company develops efficiency solutions for the heavy industry.
    * [Telecom Plan Classification Model](https://github.com/giova22i/introML): Build machine learning model to identify the right plan for each subscriber based on their behavior, using the historical data available. 
    * [Bank Customer Churn Prediction Model](https://github.com/giova22i/supervisedlearning): Creating a classification model to predict customer churn for a bank from an imbalanced dataset.
    * [Oil Well Regression Model](https://github.com/giova22i/MLB): Build a Linear Regression model that will help to pick the region with the highest profit margin. Analyze potential profit and risks using the Bootstrapping technique.
   


*Tools: sklearn, Pandas, Seaborn, Matplotlib, TensorFlow, PIL (Python Imaging Library)*

* ### **Statistical Data Analysis and Visualisation (Python)**
    *  [Video Game Market Analysis](https://github.com/giova22i/Video-Game-Sales-Analysis): Data Analysis identifying what factors make a video game succeed.  Identify patterns in historical game sales data, analyze metrics for each video game platform, and conduct statistical hypothesis testing to find potential big winners and plan advertising campaigns.
    * [Telecom Plan Analysis](https://github.com/giova22i/Prepaid-Plan-Analysis): Preliminary analysis of the plans based on a relatively small client selection. Analyze clients' behavior and determine which prepaid plan brings in more revenue. Conduct statistical hypothesis testing on profit from different plan users and different regions.
    * [Vehicle Sales Analysis](https://github.com/giova22i/Practicum-EDA): Analysis on what factors affect the price of a vehicle to be listed on a car sales website.

*Tools: Pandas, Seaborn and Matplotlib, SciPy*

* ### **Data Collection and Storage (Python and PostgreSQL)**
    * [Ride Sharing App Analysis](https://github.com/giova22i/SQL-Project): Data Analysis on Chicago taxicab rides and weather reports to advise hypothetical ride-sharing company Zuber. Study a database, analyze data from competitors, and test hypothesis about the impact of weather on ride frequency.
    
*Tools: Beautiful Soup, Requests, Pandas, PostgreSQL, SciPy.stats,NumPy*

* ### __Data Preprocessing (Python)__
   * [Credit Score Analysis](https://github.com/giova22i/Credit-Report-Analysis): Analyzing borrowers’ risk of defaulting. Prepare a report for a bank’s loan division to determine the likelihood that a customer defaults on a loan. Find out if a customer’s marital status and number of children has an impact on whether they will default on a loan. 

*Tools: Python, NLTK, WordNetLemmatizer, SnowballStemmer*

##  <p align=center> Undergraduate Research Projects (University at Buffalo)  </p>
* ### __Quantitative Research Analysis__   
    * [Ethnic-Identity-Development-Research-Project](https://github.com/giova22i/Ethnic-Identity-Development-Research-Project):  Research of ethnic identity development by examining how three components of ethnic identity: ethnic search, achievement and commitment; change across the transition from high school to college.
    * [Effect of Anxiety on Self Estimates of Intelligence](https://github.com/giova22i/Anxiety-Effect-on-SEI): Determine if higher levels of state anxiety would lead to lower self-estimates of intelligence among undergraduate students.
    * [Advanced Psych Research Methods Research Project](https://github.com/giova22i/Psych-Research-Project): Experimental analysis to examine the relationship between phone use and the amount of sleep acquired

Tools: *SPSS Statistics*

__Psychological Assessment Scales and Measures__:
   * *Multigroup Ethnic Identity Measure (MEIM)*
   * *State-Trait Anxiety Inventory (STAI)*




