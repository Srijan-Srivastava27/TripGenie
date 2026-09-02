# Importing Libraries

import pandas as pd
import numpy as np
import pickle # for saving the trained model and label encoders to a file

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity # for similarity calculation between 2 users
from sklearn.feature_extraction.text import TfidfVectorizer # for converting text data into numericals representation for better understanding of machine learning algorithms
import matplotlib.pyplot as plt # for plotting graphs and data visiulization
import seaborn as sns # for plotting graphs and data visiulization
from sklearn.model_selection import train_test_split # for splitting the dataset into training and testing data
from sklearn.ensemble import RandomForestRegressor # for training the model
from sklearn.preprocessing import LabelEncoder # for encoding categorical data into numerical data
from sklearn.metrics import mean_squared_error, r2_score # for calculating the accuracy of the model and mean squared error

#LOADING DATASETS AND READING CSV FILES INTO DATAFRAMES
destinations_df = pd.read_csv('D:\Travel Recommidation System\code and dataset\Expanded_Destinations.csv') # info about various destinations in india including details like type of destination
review_df = pd.read_csv("D:\Travel Recommidation System\code and dataset\Final_Updated_Expanded_Reviews.csv") # info about reviews given by users to the destinations including details like rating, review, user id, destination id
userhistory_df = pd.read_csv("D:\Travel Recommidation System\code and dataset\Final_Updated_Expanded_UserHistory.csv") # info about user history including details like user id, destination id, date of visit, duration of stay, etc.
user_df = pd.read_csv("D:\Travel Recommidation System\code and dataset\Final_Updated_Expanded_Users.csv") # Profiles of users including their preferences and includes details on the number of adults and children for travel.

#DATA PREPROCESSING
# Dropping unnecessary columns from the datasets
reviews_destinations = pd.merge(review_df, destinations_df, on='DestinationID', how='inner')# Merge reviews and destinations dataframes on 'DestinationID'
reviews_destination_userhistory = pd.merge(reviews_destinations, userhistory_df, on='UserID', how='inner')# Merge reviews_destinations and userhistory_df dataframes on 'UserID'
df = pd.merge(reviews_destination_userhistory, user_df, on='UserID', how='inner')# Merge reviews_destination_userhistory and user_df dataframes on 'UserID'

#content based recommendation
df['features'] = df['Type'] + " " + df['State'] + " " + df['BestTimeToVisit'] + " " + df['Preferences'] 
vectorizer = TfidfVectorizer(stop_words='english') # Convert text data into array of numbers using TF-IDF Vectorizer
destination_features = vectorizer.fit_transform(df['features']) # Convert the features into a matrix of TF-IDF features
cosine_sim = cosine_similarity(destination_features, destination_features) # Compute the cosine similarity matrix and compare the similarity between them

#define recommendation function 
def recommend_destinations(user_id, userhistory_df, destination_df, cosine_sim): # Recommends top 5 destionations to the user based on the user history and cosine similarity
    visited_destinations = userhistory_df[userhistory_df['UserID'] == user_id]['DestinationID'].values # filtering to Get the list of destinations visited by the user
    similar_scores = np.sum(cosine_sim[visited_destinations], axis=0) # Get the similarity scores for the visited destinations
    recommended_destinations_idx = np.argsort(similar_scores)[::-1] # Sort the indices of the destinations based on the similarity scores in descending order 
    recommendations = [] # Initialize an empty list to store the recommended destinations
    for idx in recommended_destinations_idx: # Loop through the sorted indices
        if destinations_df.iloc[idx]['DestinationID'] not in visited_destinations:
            recommendations.append(destination_df.iloc[idx][['DestinationID', 'Name', 'State', 'Type', 'Popularity', 'BestTimeToVisit']].to_dict()) # Append the recommended destination to the list i.e. top recommended destinations
        if len(recommendations) >= 5: # If the list has 5 recommended destinations, break the loop
            break
    return pd.DataFrame(recommendations) # Convert the list of recommended destinations to a DataFrame and return it
    
#recommended_destinations = recommend_destinations(20, userhistory_df, destinations_df, cosine_sim) # Call the recommendation function with user id 1 and get the recommended destinations
#print(recommended_destinations) # Print the recommended destinations

#Collaborative filtering i.e. recommendation based on user preferences and ratings
user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating').fillna(0) # Create a user-item matrix with users as rows and destinations as columns, filling NaN values with 0
user_similarity = cosine_similarity(user_item_matrix) # Compute the cosine similarity matrix between users

def collaborative_recommend(user_id, user_similarity, user_item_matrix, destination_df): # find similar users
    similar_users = user_similarity[user_id - 1] #finding the similar users for the given user id
    similar_users_idx = np.argsort(similar_users)[::-1][1:6] # Get the indices of the top 5 similar users
    similar_users_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0) # Get the destinations liked by similar users
    recommended_destination_ids = similar_users_ratings.sort_values(ascending=False).head(5).index # Get the top 5 recommended destination IDs
    recommendations = destination_df[destination_df['DestinationID'].isin(recommended_destination_ids)][['DestinationID', 'Name', 'State', 'Type', 'Popularity', 'BestTimeToVisit']] # Filter the destination DataFrame to get the detailed recommended destinations
    return recommendations
    
#collaborative_recommendations = collaborative_recommend(15,  user_similarity, user_item_matrix, destinations_df) # Call the recommendation function with user id 1 and get the recommended destinations
#print(collaborative_recommendations) # Print the recommended destinations

#User input feature base recommendation
data = df.copy() # Create a copy of the DataFrame to avoid modifying the original data

# select relevant features
features = ['Name_x', 'State', 'Type', 'BestTimeToVisit', 'Preferences', 'Gender', 'NumberOfAdults', 'NumberOfChildren'] 
target = 'Popularity' # Predicting Popularity based on the features

#Encoding
label_encoder = {} # Initialize an empty dictionary to store label encoders for each feature

for col in features:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoder[col] = le
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split the data into training and testing sets in 80:20 ratio

#Model Training
model = RandomForestRegressor(random_state=42) # Initialize the Random Forest Regressor model
model.fit(X_train, y_train) # Fit the model to the training data
y_pred = model.predict(X_test) # Predict the target variable on the test data
#print("Mean Squared Error:", mean_squared_error(y_test, y_pred)) # Print the Mean Squared Error of the model

def recommend_destinations(user_input, model, label_encoders, features, data):
    # Encode user input using the label encoders
    encoded_input = {}
    for feature in features:
        if feature in label_encoders:
            encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
        else:
            encoded_input[feature] = user_input[feature]
    
    # Convert user input to DataFrame
    input_df = pd.DataFrame([encoded_input])
    
    # Predict popularity using the trained model
    predicted_popularity = model.predict(input_df)[0]
    
    return predicted_popularity


# Example user input
user_input = {
    'Name_x': 'Jaipur City',
    'State': 'Rajasthan',
    'Type': 'City',
    'BestTimeToVisit': 'Oct-Mar',
    'Preferences': 'City, Historical',
    'Gender': 'Female',
    'NumberOfAdults': 2,
    'NumberOfChildren': 1
}

# Make prediction
predicted_popularity = recommend_destinations(user_input, model, label_encoder, features, data)  # Call the recommendation function with user input and get the predicted popularity
print(f"Predicted Popularity: {predicted_popularity:.2f}")  # Print the predicted popularity of the destination based on user input

# Save the trained model and label encoders
pickle.dump(model, open('model.pkl', 'wb'))  # Save the trained model to a file using pickle
pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))  # Save the label encoders to a file using pickle

# Save the DataFrame to the "code and dataset" folder
df.to_csv('D:/Travel Recommidation System/code and dataset/final_df.csv', index=False)

print("DataFrame saved successfully!")




