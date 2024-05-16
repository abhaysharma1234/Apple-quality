#!/usr/bin/env python
# coding: utf-8

# # **Project Name**    - 
# Apple Quality
# 

# ##### **Project Type**    - Classification
# ##### **Contribution**    - Team
# ##### **Team Member 1 - Abhay Sharma(2210990028)
# ##### **Team Member 2 - Agrim(2210990079)
# ##### **Team Member 3 - Ankush(2210990119)
# ##### **Team Member 4 - Swastik chauhan(2210992431)

# # **Project Summary -**

# In this project, we aimed to develop a machine learning model to predict the quality of apples based on various features such as sweetness, acidity, firmness, and color. The quality of fruits is a critical factor for both producers and consumers, and having a reliable predictive model can assist in quality control and decision-making processes.The dataset used for this project consists of observations collected from apple farms, including measurements of different attributes of apples and their corresponding quality ratings. The dataset includes features such as sweetness level, acidity level, firmness, color intensity, and the quality rating assigned to each apple. It contains both numerical and categorical data, making it suitable for classification tasks.Before building the predictive model, we performed several preprocessing steps on the dataset. This included handling missing values, encoding categorical variables, and scaling numerical features. Missing values were imputed using appropriate techniques such as mean or median imputation. Categorical variables were encoded using label encoding or one-hot encoding based on their nature. Numerical features were scaled using standardization to ensure that all features contribute equally to the model.
# 
# 

# # **GitHub Link -**

# Provide your GitHub Link here.

# # **Problem Statement**
# 

# The main objective is to build a predictive model, which could help in predicting apple's quality.

# # **General Guidelines** : -  

# 1.   Well-structured, formatted, and commented code is required. 
# 2.   Exception Handling, Production Grade Code & Deployment Ready Code will be a plus. Those students will be awarded some additional credits. 
#      
#      The additional credits will have advantages over other students during Star Student selection.
#        
#              [ Note: - Deployment Ready Code is defined as, the whole .ipynb notebook should be executable in one go
#                        without a single error logged. ]
# 
# 3.   Each and every logic should have proper comments.
# 4. You may add as many number of charts you want. Make Sure for each and every chart the following format should be answered.
#         
# 
# ```
# # Chart visualization code
# ```
#             
# 
# *   Why did you pick the specific chart?
# *   What is/are the insight(s) found from the chart?
# * Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.
# 
# 5. You have to create at least 15 logical & meaningful charts having important insights.
# 
# 
# [ Hints : - Do the Vizualization in  a structured way while following "UBM" Rule. 
# 
# U - Univariate Analysis,
# 
# B - Bivariate Analysis (Numerical - Categorical, Numerical - Numerical, Categorical - Categorical)
# 
# M - Multivariate Analysis
#  ]
# 
# 
# 
# 
# 
# 6. You may add more ml algorithms for model creation. Make sure for each and every algorithm, the following format should be answered.
# 
# 
# *   Explain the ML Model used and it's performance using Evaluation metric Score Chart.
# 
# 
# *   Cross- Validation & Hyperparameter Tuning
# 
# *   Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.
# 
# *   Explain each evaluation metric's indication towards business and the business impact pf the ML model used.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# # ***Let's Begin !***

# ## ***1. Know Your Data***

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# warnings.filterwarnings('ignore')


# ### Dataset Loading

# In[2]:


# Load Data
apple = pd.read_csv('C:/Users/abhay/Downloads/apple_quality.csv')


# ### Dataset First View

# In[3]:


# Dataset First Look
apple.head()


# ### Dataset Rows & Columns count

# In[4]:


# Dataset Rows & Columns count
apple.shape


# ### Dataset Information

# In[5]:


# Dataset Info
apple.info()


# #### Duplicate Values

# In[6]:


# Dataset Duplicate Value Count
duplicate_count = apple.duplicated().sum()

duplicate_count


# #### Missing Values/Null Values

# In[7]:


# Missing Values/Null Values Count
apple.isnull().sum()


# In[8]:


# Visualizing the missing values
plt.figure(figsize = (20,7))
sns.heatmap(apple.isnull(), cbar=False)


# ### What did you know about your dataset?

# This dataset have rows 4000 and 9 columns with no duplicated values.Dataset have 8 int64 data_type columns and 1 object data type column.Columns like weight,sweetness will help in predicting apple's quality.

# ## ***2. Understanding Your Variables***

# In[9]:


# Dataset Columns
apple.columns


# In[10]:


# Dataset Describe
apple.describe()


# ### Variables Description 

# As mentioned earlier dataset has 8 int columns.So,we can see the max and min values of columns with 25,50,75 percentile.

# ### Check Unique Values for each variable.

# In[11]:


# Check Unique Values for each variable.
unique = apple.nunique().reset_index()
unique


# ## 3. ***Data Wrangling***

# ### Data Wrangling Code

# In[12]:


# Write your code to make your dataset analysis ready.
apple.isnull().sum()


# In[13]:


apple.groupby(['Weight','Sweetness']).mean().reset_index()


# In[14]:


apple.groupby(['Crunchiness', 'Juiciness'])['Ripeness'].mean()


# ### What all manipulations have you done and insights you found?

# 1.We have no null values in this data.
# 
# 2.Overall quality of apples is good.
# 
# 3.Most of the apples are good which have high sweetness value.
# 
# 4.Most of the apples are good which have high juiciness value.
# 
# 5.Maximum apples have high sweetness value.

# ## ***4. Data Vizualization, Storytelling & Experimenting with charts : Understand the relationships between variables***

# #### Chart - 1

# In[15]:


# Chart - 1 visualization code
acidity_data = apple['Acidity']

# Plotting the box plot
plt.figure(figsize=(8, 6))
plt.boxplot(acidity_data, vert=False)
plt.title('Box Plot of Acidity')
plt.xlabel('Acidity')
plt.show()


# ##### 1. Why did you pick the specific chart?

# I chose box plots to visualize acidity variation across quality ratings due to their ability to display distribution and outliers effectively.

# ##### 2. What is/are the insight(s) found from the chart?

# From graph we can see that most of values occur in between -2 and 2.

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# Yes, the insights can create a positive business impact by guiding quality control efforts and product development strategies. However, a potential negative impact could arise if the weak positive correlation between sweetness and acidity leads to an undesirable trade-off between these attributes, affecting consumer preferences and market acceptance of apple products.

# #### Chart - 2

# In[16]:


# Chart - 2 visualization code
plt.figure(figsize=(10, 6))
plt.hist(apple['Crunchiness'], bins=10)
plt.xlabel('Crunchiness')
plt.ylabel('Frequency')
plt.title('Histogram of Crunchiness')
plt.show()


# ##### 1. Why did you pick the specific chart?

# Histograms are effective for visualizing the distribution of a single continuous variable.

# ##### 2. What is/are the insight(s) found from the chart?

# This plot can show the frequency distribution of Crunchiness levels.

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# Understanding the variability in weights can inform quality control measures and potentially lead to improvements in product consistency, enhancing customer satisfaction and loyalty.

# #### Chart - 3

# In[17]:


# Chart - 3 visualization code
plt.figure(figsize=(10, 6))
plt.scatter(apple['Sweetness'], apple['Juiciness'])
plt.xlabel('Sweetness')
plt.ylabel('Juiciness')
plt.title('Scatterplot of Sweetness vs Juiciness')
plt.show()


# ##### 1. Why did you pick the specific chart?

# Scatterplots are ideal for visualizing the relationship between two continuous variables.
# 

# ##### 2. What is/are the insight(s) found from the chart?

# This plot can help identify any correlation or patterns between sweetness and juiciness.

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# Understanding the relationship between sweetness and juiciness can guide product development efforts, ensuring that products meet consumer preferences, potentially leading to increased sales and market share.

# #### Chart - 4

# In[18]:


# Chart - 4 visualization code
plt.figure(figsize=(10, 6))
plt.hist(apple['Sweetness'], bins=10)
plt.xlabel('Sweetness')
plt.ylabel('Frequency')
plt.title('Histogram of Sweetness')
plt.show()


# ##### 1. Why did you pick the specific chart?

# Histograms are effective for visualizing the distribution of a single continuous variable.

# ##### 2. What is/are the insight(s) found from the chart?

# This plot can show the frequency distribution of sweetness levels

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# Understanding the acidity distribution can inform product formulation and flavor profiles, ensuring that products meet consumer preferences, potentially leading to increased sales and customer satisfaction.

# #### Chart - 5

# In[19]:


# Chart - 5 visualization code
apple['Quality'].value_counts().plot(kind='bar')
plt.xlabel('Quality')
plt.ylabel('Frequency') 
plt.title('Distribution of Quality')
plt.show()


# ##### 1. Why did you pick the specific chart?

# I have used Bar plot because they are commonly used for visualizing categorical data as they show the frequency or count of each category using bars

# ##### 2. What is/are the insight(s) found from the chart?

# Ratio of good and bad is almost equal.

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# From the graph we can see that good and bad are almost equal.

# #### Chart - 6

# In[20]:


# Chart - 6 visualization code
plt.figure(figsize=(10, 6))
plt.scatter(apple['Crunchiness'], apple['Juiciness'])
plt.xlabel('Crunchiness')
plt.ylabel('Juiciness')
plt.title('Scatterplot of Crunchiness vs Juiciness')
plt.show()


# ##### 1. Why did you pick the specific chart?

# Scatterplots are ideal for visualizing the relationship between two continuous variables.

# ##### 2. What is/are the insight(s) found from the chart?

# This plot can help identify any correlation or patterns between Crunchiness and juiciness.

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# Understanding the relationship between Crunchiness and juiciness can guide product development efforts, ensuring that products meet consumer preferences, potentially leading to increased sales and market share.

# #### Chart - 7

# In[21]:


# Chart - 7 visualization code
plt.figure(figsize=(10, 6))
ap = apple.head(40)
plt.scatter(ap['Juiciness'], ap['Acidity'])
plt.xlabel('Juiciness')
plt.ylabel('Acidity')
plt.title('Scatterplot of Ripeness vs Acidity')
plt.show()


# ##### 1. Why did you pick the specific chart?

# Scatterplots are ideal for visualizing the relationship between two continuous variables.

# ##### 2. What is/are the insight(s) found from the chart?

# This plot can help identify any correlation or patterns between Ripeness and Acidity.

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# Understanding the relationship between Crunchiness and juiciness can guide product development efforts, ensuring that products meet consumer preferences, potentially leading to increased sales and market share

# #### Chart - 8

# In[22]:


# Chart - 8 visualization code
plt.figure(figsize=(10, 6))
plt.hist(apple['Ripeness'], bins=10)
plt.xlabel('Ripeness')
plt.ylabel('Frequency')
plt.title('Histogram of Ripeness')
plt.show()


# ##### 1. Why did you pick the specific chart?

# Histograms are effective for visualizing the distribution of a single continuous variable.

# ##### 2. What is/are the insight(s) found from the chart?

# This plot can show the frequency distribution of Ripeness levels

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# Understanding the acidity distribution can inform product formulation and flavor profiles, ensuring that products meet consumer preferences, potentially leading to increased sales and customer satisfaction.

# #### Chart - 9

# In[23]:


# Chart - 9 visualization code
plt.figure(figsize=(10, 6))
plt.pie(apple['Quality'].value_counts(), labels=apple['Quality'].unique(), autopct='%1.1f%%')
plt.title('Piechart of Quality')
plt.show()


# ##### 1. Why did you pick the specific chart?

# Pie charts are an effective way to visualize the distribution of categorical data. In this case, the quality column represents categorical data with different quality levels. A pie chart allows us to see the proportion of each quality level relative to the whole dataset, making it easy to understand the distribution at a glance.

# ##### 2. What is/are the insight(s) found from the chart?

# The pie chart illustrates the distribution of quality levels among the items in the dataset. By looking at the proportions of each quality level, we can identify which quality levels are most common and which are less prevalent. This insight can help in understanding the overall quality profile of the items and identifying any potential areas for improvement.

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# Understanding the distribution of quality levels can have several positive business impacts. For example, if a particular quality level is found to be significantly more common than others, it may indicate a strong selling point or a feature that resonates well with customers. On the other hand, if certain quality levels are less common, it may highlight areas for improvement in product quality or production processes. Addressing these insights can lead to enhanced customer satisfaction, improved brand reputation, and increased sales.

# #### Chart - 10

# In[24]:


# Chart - 10 visualization code
plt.figure(figsize=(10, 6))
plt.hist(apple['Acidity'], bins=10)
plt.xlabel('Acidity')
plt.ylabel('Frequency')
plt.title('Histogram of Acidity')
plt.show()


# ##### 1. Why did you pick the specific chart?

# Histograms are effective for visualizing the distribution of a single continuous variable.

# ##### 2. What is/are the insight(s) found from the chart?

# This plot can show the frequency distribution of Acidity levels

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# Understanding the acidity distribution can inform product formulation and flavor profiles, ensuring that products meet consumer preferences, potentially leading to increased sales and customer satisfaction.

# #### Chart - 11

# In[25]:


# Chart - 11 visualization code


# ##### 1. Why did you pick the specific chart?

# Answer Here.

# ##### 2. What is/are the insight(s) found from the chart?

# Answer Here

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# Answer Here

# #### Chart - 12

# In[26]:


# Chart - 12 visualization code


# ##### 1. Why did you pick the specific chart?

# Answer Here.

# ##### 2. What is/are the insight(s) found from the chart?

# Answer Here

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# Answer Here

# #### Chart - 13

# In[27]:


# Chart - 13 visualization code


# ##### 1. Why did you pick the specific chart?

# Answer Here.

# ##### 2. What is/are the insight(s) found from the chart?

# Answer Here

# ##### 3. Will the gained insights help creating a positive business impact? 
# Are there any insights that lead to negative growth? Justify with specific reason.

# Answer Here

# #### Chart - 14 - Correlation Heatmap

# In[28]:


# Correlation Heatmap visualization code
plt.figure(figsize=(12,6))
sns.heatmap(apple.corr() , annot = True)


# ##### 1. Why did you pick the specific chart?

# I chose to create a correlation heatmap because it provides a clear visual representation of the correlation between different variables in the dataset.

# ##### 2. What is/are the insight(s) found from the chart?

# The correlation between Weight and Sweetiness is negative.This means that apples with high sweetness will tend to have less acidity and vice versa.  However, the negative correlation is weaker than the positive correlation between Sweetness and Crunchiness.

# #### Chart - 15 - Pair Plot 

# In[29]:


# Pair Plot visualization code
sns.pairplot(apple)
plt.show()


# ##### 1. Why did you pick the specific chart?

# I chose to create a pair plot because it's a comprehensive way to visualize the relationships between multiple variables in a dataset.

# ##### 2. What is/are the insight(s) found from the chart?

# The insight found from the chart is that there is a strong correlation between the Crunchiness and Sweetness. This means that apples which has high sweetness will tend to have high crunchiness as well.

# ## ***5. Hypothesis Testing***

# ### Based on your chart experiments, define three hypothetical statements from the dataset. In the next three questions, perform hypothesis testing to obtain final conclusion about the statements through your code and statistical testing.

# Answer Here.

# ### Hypothetical Statement - 1

# #### 1. State Your research hypothesis as a null hypothesis and alternate hypothesis.

# Answer Here.

# #### 2. Perform an appropriate statistical test.

# In[30]:


# Perform Statistical Test to obtain P-Value


# ##### Which statistical test have you done to obtain P-Value?

# Answer Here.

# ##### Why did you choose the specific statistical test?

# Answer Here.

# ### Hypothetical Statement - 2

# #### 1. State Your research hypothesis as a null hypothesis and alternate hypothesis.

# Answer Here.

# #### 2. Perform an appropriate statistical test.

# In[31]:


# Perform Statistical Test to obtain P-Value


# ##### Which statistical test have you done to obtain P-Value?

# Answer Here.

# ##### Why did you choose the specific statistical test?

# Answer Here.

# ### Hypothetical Statement - 3

# #### 1. State Your research hypothesis as a null hypothesis and alternate hypothesis.

# Answer Here.

# #### 2. Perform an appropriate statistical test.

# In[32]:


# Perform Statistical Test to obtain P-Value


# ##### Which statistical test have you done to obtain P-Value?

# Answer Here.

# ##### Why did you choose the specific statistical test?

# Answer Here.

# ## ***6. Feature Engineering & Data Pre-processing***

# ### 1. Handling Missing Values

# In[33]:


# Handling Missing Values & Missing Value Imputation
apple.isnull().sum().reset_index()


# #### What all missing value imputation techniques have you used and why did you use those techniques?

# We do not have any missing values.

# ### 2. Handling Outliers

# In[34]:


# Handling Outliers & Outlier treatments
apple.info()


# In[35]:


numerical_cols = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']

plt.figure(figsize=(12, 8))
apple[numerical_cols].boxplot(vert=False)
plt.title('Box Plot of Numerical Columns')
plt.xlabel('Value')
plt.show()


# In[36]:


def remove_outliers_iqr_except_quality(df):
   
    Q1 = df.drop(columns=['Quality']).quantile(0.25)
    Q3 = df.drop(columns=['Quality']).quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    for column in df.drop(columns=['Quality']).columns:
        df = df[(df[column] >= lower_bound[column]) & (df[column] <= upper_bound[column])]
    return df


df_no_outliers_except_quality = remove_outliers_iqr_except_quality(apple)


# ##### What all outlier treatment techniques have you used and why did you use those techniques?

# The outlier treatment techniques used include Interquartile Range (IQR) method for removing outliers, chosen for its robustness and simplicity in identifying extreme values.

# ### 3. Categorical Encoding

# In[37]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

categorical_cols = ['Quality']

for col in categorical_cols:
    apple[col] = label_encoder.fit_transform(apple[col])
                                             
print(apple)


# In[38]:


apple.head(5)


# #### What all categorical encoding techniques have you used & why did you use those techniques?

# The categorical encoding techniques used include Label Encoding and One-Hot Encoding. Label Encoding was chosen for its simplicity in transforming categorical data into numerical labels, while One-Hot Encoding was employed to convert categorical variables into binary vectors, preserving distinct categories without introducing ordinality.

# ### 4. Textual Data Preprocessing 
# (It's mandatory for textual dataset i.e., NLP, Sentiment Analysis, Text Clustering etc.)

# #### 1. Expand Contraction

# In[39]:


# Expand Contraction


# #### 2. Lower Casing

# In[40]:


# Lower Casing


# #### 3. Removing Punctuations

# In[41]:


# Remove Punctuations


# #### 4. Removing URLs & Removing words and digits contain digits.

# In[42]:


# Remove URLs & Remove words and digits contain digits


# #### 5. Removing Stopwords & Removing White spaces

# In[43]:


# Remove Stopwords


# In[44]:


# Remove White spaces


# #### 6. Rephrase Text

# In[45]:


# Rephrase Text


# #### 7. Tokenization

# In[46]:


# Tokenization


# #### 8. Text Normalization

# In[47]:


# Normalizing Text (i.e., Stemming, Lemmatization etc.)


# ##### Which text normalization technique have you used and why?

# Answer Here.

# #### 9. Part of speech tagging

# In[48]:


# POS Taging


# #### 10. Text Vectorization

# In[49]:


# Vectorizing Text


# ##### Which text vectorization technique have you used and why?

# Answer Here.

# ### 4. Feature Manipulation & Selection

# #### 1. Feature Manipulation

# In[50]:


# Manipulate Features to minimize feature correlation and create new features


# #### 2. Feature Selection

# In[ ]:





# ##### What all feature selection methods have you used  and why?

# Answer Here.

# ##### Which all features you found important and why?

# Answer Here.

# ### 5. Data Transformation

# #### Do you think that your data needs to be transformed? If yes, which transformation have you used. Explain Why?

# In[51]:


# Transform Your data


# ### 6. Data Scaling

# In[52]:


# Scaling your data
apple.isnull().sum()


# In[53]:


apple.describe()


# In[54]:


numerical_cols = apple.select_dtypes(include=['int64', 'float64']).columns
infinity_values = apple[numerical_cols][apple[numerical_cols] == np.inf].sum()
print("\nInfinity Values:")
print(infinity_values)


# In[55]:


x = apple.drop(columns = 'Quality')
y = apple['Quality']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[56]:


scaler.fit_transform(x)


# In[57]:


print(apple)


# ##### Which method have you used to scale you data and why?

# In[ ]:





# ### 7. Dimesionality Reduction

# ##### Do you think that dimensionality reduction is needed? Explain Why?

# Answer Here.

# In[58]:


# DImensionality Reduction (If needed)


# ##### Which dimensionality reduction technique have you used and why? (If dimensionality reduction done on dataset.)

# Answer Here.

# ### 8. Data Splitting

# In[ ]:





# In[ ]:





# ##### What data splitting ratio have you used and why? 

# 

# ### 9. Handling Imbalanced Dataset

# ##### Do you think the dataset is imbalanced? Explain Why.

# Answer Here.

# In[59]:


# Handling Imbalanced Dataset (If needed)


# ##### What technique did you use to handle the imbalance dataset and why? (If needed to be balanced)

# Answer Here.

# ## ***7. ML Model Implementation***

# ### ML Model - 1

# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


X = apple.drop('Quality', axis=1)  
y = apple['Quality']  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

logistic_model = LogisticRegression(max_iter=1000)  
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# #### 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.

# In[61]:


# Visualizing evaluation Metric Score chart


# #### 2. Cross- Validation & Hyperparameter Tuning

# In[62]:


# ML Model - 1 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

# Fit the Algorithm

# Predict on the model


# ##### Which hyperparameter optimization technique have you used and why?

# Answer Here.

# ##### Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.

# Answer Here.

# ### ML Model - 2

# In[63]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# #### 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.

# In[64]:


# Visualizing evaluation Metric Score chart


# #### 2. Cross- Validation & Hyperparameter Tuning

# In[65]:


# ML Model - 1 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

# Fit the Algorithm

# Predict on the model


# ##### Which hyperparameter optimization technique have you used and why?

# Answer Here.

# ##### Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.

# Answer Here.

# #### 3. Explain each evaluation metric's indication towards business and the business impact pf the ML model used.

# Answer Here.

# ### ML Model - 3

# In[66]:


# ML Model - 3 Implementation

# Fit the  svm

# Predict on the model

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


X = apple.drop('Quality', axis=1) 
y = apple['Quality']  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='rbf', C=1.0)  
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# #### 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.

# In[67]:


# Visualizing evaluation Metric Score chart


# #### 2. Cross- Validation & Hyperparameter Tuning

# In[68]:


# ML Model - 3 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

# Fit the Algorithm

# Predict on the model


# ##### Which hyperparameter optimization technique have you used and why?

# Answer Here.

# ##### Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.

# Answer Here.

# ### 1. Which Evaluation metrics did you consider for a positive business impact and why?

# Answer Here.

# ### 2. Which ML model did you choose from the above created models as your final prediction model and why?

# Answer Here.

# ### 3. Explain the model which you have used and the feature importance using any model explainability tool?

# Answer Here.

# ## ***8.*** ***Future Work (Optional)***

# ### 1. Save the best performing ml model in a pickle file or joblib file format for deployment process.
# 

# In[69]:


# Save the File


# ### 2. Again Load the saved model file and try to predict unseen data for a sanity check.
# 

# In[70]:


# Load the File and predict unseen data.


# ### ***Congrats! Your model is successfully created and ready for deployment on a live server for a real user interaction !!!***

# # **Conclusion**

# Write the conclusion here.

# ### ***Hurrah! You have successfully completed your Machine Learning Capstone Project !!!***
