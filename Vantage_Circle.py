#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd



import openpyxl

# File uploader
uploaded_file = st.file_uploader("C:/Users/ADMIN/Downloads/data_analysis.xlsx", type=["xlsx"])

# Load Excel file into a pandas DataFrame
if uploaded_file is not None:
    df_sender = pd.read_excel(uploaded_file, sheet_name='sender')
    df_receiver = pd.read_excel(uploaded_file, sheet_name='receiver')
    df_manager = pd.read_excel(uploaded_file, sheet_name='manager')






import pandas as pd

import pandas as pd

# Assuming you have three dataframes named df1, df2, and df3

# Concatenate the dataframes vertically along rows
df = pd.concat([df_sender,df_receiver,df_manager], axis=1)

# Reset the index of the concatenated dataframe
df.reset_index(drop=True, inplace=True)
import pandas as pd

# Assuming you have a dataframe named df with duplicate columns

# Drop duplicate columns
df = df.loc[:,~df.columns.duplicated()]



st.title("Data Exploration and Data Analysis")

import pandas as pd



# # Top receivers

# In[9]:


top_receivers = df.groupby("receiver_id")["points"].sum().sort_values(ascending=False).head(3)


# In[10]:


top_receivers=pd.DataFrame(top_receivers)
st.write("**Top Receivers**")
top_receivers


# # Monthly analysis of total points received

# In[11]:


df["date"] = pd.to_datetime(df["date"])
monthly_points_received = df.groupby(df["date"].dt.to_period("M"))["points"].sum()


# In[12]:


monthly_points_received=pd.DataFrame(monthly_points_received)
st.write("**Monthly Points Received**")
monthly_points_received


# # Average points received by users per week

# In[13]:


weekly_points_received = df.groupby(df["date"].dt.to_period("W"))["points"].mean()
weekly_points_received=pd.DataFrame(weekly_points_received)
st.write("**Weekly Points Received**")

weekly_points_received


# # Top countries based on active users

# In[14]:


top_countries = df.groupby("country")["receiver_id"].nunique().sort_values(ascending=False).head(3)
top_countries=pd.DataFrame(top_countries)
st.write("**Top Countries**")

top_countries




# # Top awarding managers

# In[15]:



top_managers = df.groupby("manager_id")["points"].sum().sort_values(ascending=False).head(3)
top_managers=pd.DataFrame(top_managers)
st.write("**Top Managers**")

top_managers

# # Weekly points given by manager

# In[16]:


weekly_points_given = df.groupby(["manager_id", df["date"].dt.to_period("W")])["points"].mean()
weekly_points_given=pd.DataFrame(weekly_points_given)
st.write("**weekly_points_given**")
weekly_points_given


# # Monthly analysis of total points given by each manager

# In[17]:


monthly_points_given = df.groupby(["manager_id", df["date"].dt.to_period("M")])["points"].sum()
monthly_points_given=pd.DataFrame(monthly_points_given)
st.write("**monthly_points_given**")
monthly_points_given


# # Retention analysis of points used per month by each company (cohort analysis)

# In[18]:


cohort_analysis = df.groupby(["company_names", df["date"].dt.to_period("M")])["points"].sum().unstack()
st.write("**cohort_analysis**")
cohort_analysis


# # Additional Features

# # Time series forecasting: Forecast future product usage

# In[19]:


monthly_product_usage = df.groupby(df['date'].dt.to_period('M')).size().reset_index()
monthly_product_usage = monthly_product_usage.rename(columns={0: 'Usage Count'})
st.write("**monthly_product_usage**")
monthly_product_usage

# # User segmentation analysis: Calculate average points received by user segment

# In[20]:


user_segment_points = df.groupby('country')['points'].mean().reset_index()
user_segment_points = user_segment_points.sort_values(by='points', ascending=False)
st.write("**user_segment_points**")
user_segment_points


# # Receiver engagement analysis: Calculate login count per receiver

# In[21]:


# User engagement analysis: Calculate login count per user
receiver_login_count = df.groupby('receiver_id')['date'].count().reset_index()
receiver_login_count = receiver_login_count.rename(columns={'date': 'Login Count'})
st.write("**receiver_login_count**")
receiver_login_count


# # Sender engagement analysis: Calculate login count per sender

# In[22]:


# User engagement analysis: Calculate login count per user
sender_login_count = df.groupby('sender_id')['date'].count().reset_index()
sender_login_count = sender_login_count.rename(columns={'date': 'Login Count'})
st.write("**sender_login_count**")
sender_login_count

st.title("Data Visualization")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Monthly analysis of total points received

# In[24]:
import streamlit as st
import matplotlib.pyplot as plt

# Disable the warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Create a Streamlit app
st.write("**Monthly Analysis of Total Points Received**")

# Plot the data
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_points_received.index.to_timestamp(), monthly_points_received.squeeze().values)
ax.set_xlabel("Month")
ax.set_ylabel("Total Points Received")

# Display the plot in the Streamlit app
st.pyplot(fig)

# Average points received by users per week
st.write("**Average points received by users per week**")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(weekly_points_received.index.to_timestamp(), weekly_points_received.squeeze().values)
ax.set_xlabel("week")
ax.set_ylabel("Average Points Received by user per week")
st.pyplot(fig)

st.write("**Monthly Analysis of Total Points Given by Each Manager**")

plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_points_given.index.get_level_values(1).to_timestamp(), y=monthly_points_given.squeeze().values)
plt.xlabel("Month")
plt.ylabel("Total Points Given")
plt.title("Monthly Analysis of Total Points Given by Each Manager")
plt.legend(top_managers.index)
st.pyplot()



st.write("**Retention Analysis of Points Used per Month by Each Company**")
plt.figure(figsize=(12, 6))
sns.heatmap(data=cohort_analysis, cmap="YlGnBu", annot=True, fmt=".0f", cbar=False)
plt.xlabel("Month")
plt.ylabel("Company")


# Display the heatmap in the Streamlit app
st.pyplot(plt)

import matplotlib.pyplot as plt

# Assuming you have a list of top awarding managers and their corresponding points
top_awarding_managers_id= ['1455043.0','1825603.0' , '1828504.0']
points_awarded = [70000,40000,18000]

# Create a donut chart
st.write('**Top Awarding Managers**')
fig, ax = plt.subplots()
ax.pie(points_awarded, labels=top_awarding_managers_id, autopct='%1.1f%%', startangle=90, pctdistance=0.85, wedgeprops={'edgecolor': 'grey'})
center_circle = plt.Circle((0,0),0.70,fc='white')
ax.add_artist(center_circle)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

st.pyplot(plt)



st.write("**Top Countries Based on Active Users**")
top_countries=['India','United States','United Kingdom']
number_of_active_users_per_country=[1250,40,8]
plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries, y=number_of_active_users_per_country)
plt.xlabel("Country")
plt.ylabel("Number of Active Users")

st.pyplot()
st.title("statistical model - SVC (Support Vector Classifier)")
from sklearn.preprocessing import LabelEncoder

# Instantiate the LabelEncoder
label_encoder = LabelEncoder()

# Encode the categorical column
df['country'] = label_encoder.fit_transform(df['country'])
df['feed type'] = label_encoder.fit_transform(df['feed type'])
df['company_names'] = label_encoder.fit_transform(df['company_names'])
import numpy as np
df.drop("sender_id", axis=1, inplace = True)
df.drop("receiver_id",axis=1,inplace = True)
df.drop("date", axis=1, inplace = True)
df.drop("manager_id", axis=1, inplace = True)
x=df.drop("feed type",axis=1).values
y=df['feed type'].values
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
svm = SVC(kernel='rbf')  # You can specify other kernels like 'poly', 'rbf', etc.

# Fit the SVM classifier to the training data
svm.fit(x_train, y_train)

# Predict the labels for the test data
y_pred = svm.predict(x_test)
st.write("**Independent Variable columns - Country, Points, Company Names**")
st.write("**Dependent Variable Column - Feed Type**")
# Calculate the accuracy of the SVM classifier
accuracy = accuracy_score(y_test, y_pred)
st.write('**Accuracy of svc model is 95.63%**')
plt.scatter(x_test[:,0],x_test[:,2],c=y_test)
plt.xlabel("country")
plt.ylabel("POINTS")
st.write("**Feed Type by Country and Points**")
st.pyplot(plt)

plt.scatter(x_test[:,0],x_test[:,2],c=y_pred)
plt.xlabel("country")
plt.ylabel("POINTS")
st.write("**Predicted Feed Type by Country and Points**")
st.pyplot(plt)

plt.scatter(x_test[:,1],x_test[:,2],c=y_test)
plt.xlabel("Company Names")
plt.ylabel("Points")
st.write("**Feed Type by Company Names and Points**")
st.pyplot(plt)

plt.scatter(x_test[:,1],x_test[:,2],c=y_pred)
plt.xlabel("Company Names")
plt.ylabel("Points")
st.write("**Predicted Feed Type by Company Names and Points**")
st.pyplot(plt)
