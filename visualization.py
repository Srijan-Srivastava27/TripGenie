
import matplotlib.pyplot as plt # for plotting graphs and data visiulization
import seaborn as sns
import pandas as pd

destinations_df = pd.read_csv('D:\Travel Recommidation System\Expanded_Destinations.csv')
review_df = pd.read_csv("D:\Travel Recommidation System\Final_Updated_Expanded_Reviews.csv")
userhistory_df = pd.read_csv("D:\Travel Recommidation System\Final_Updated_Expanded_UserHistory.csv")
user_df = pd.read_csv("D:\Travel Recommidation System\Final_Updated_Expanded_Users.csv")

#destination popularity
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Popularity', y='Name', data=destinations_df.sort_values('Popularity', ascending=False), palette='viridis')
# plt.title('Most Popular Destinations')
# plt.xlabel('Popularity')
# plt.ylabel('Destination Name')
# plt.show()

#destination types distribution
# plt.figure(figsize=(8, 6))
# sns.countplot(y='Type', data=destinations_df, order=destinations_df['Type'].value_counts().index, palette='coolwarm')
# plt.title('Distribution of Destination Types')
# plt.xlabel('Count')
# plt.ylabel('Type')
# plt.show()

# Best time to visit distribution
# plt.figure(figsize=(8, 6))
# sns.countplot(y='BestTimeToVisit', data=destinations_df, order=destinations_df['BestTimeToVisit'].value_counts().index, palette='magma')
# plt.title('Distribution of Best Time to Visit')
# plt.xlabel('Count')
# plt.ylabel('Season')
# plt.show()

#Rating distribution
# plt.figure(figsize=(8, 6))
# sns.histplot(review_df['Rating'], bins=5, kde=True, color='blue')
# plt.title('Distribution of Ratings')
# plt.xlabel('Rating')
# plt.ylabel('Frequency')
# plt.show()