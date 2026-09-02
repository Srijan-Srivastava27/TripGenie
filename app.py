from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import os
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import secrets

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database setup
def get_db_connection():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'users.db'))
    conn.row_factory = sqlite3.Row
    return conn

# Initialize the database if it doesn't exist
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fullname TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    
    # Create user_mapping table to map emails to user IDs for recommendations
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_mapping (
        email TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

# app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a random secret key

# Initialize database
init_db()

# Load datasets and models using relative paths
features = ['Name_x', 'State', 'Type', 'BestTimeToVisit', 'Preferences', 'Gender', 'NumberOfAdults', 'NumberOfChildren']

model_path = os.path.join(BASE_DIR, 'model.pkl')
label_encoders_path = os.path.join(BASE_DIR, 'label_encoder.pkl')
destinations_path = os.path.join(BASE_DIR, 'code and dataset', 'Expanded_Destinations.csv')
userhistory_path = os.path.join(BASE_DIR, 'code and dataset', 'Final_Updated_Expanded_UserHistory.csv')
df_path = os.path.join(BASE_DIR, 'code and dataset', 'Final_df.csv')

model = pickle.load(open(model_path, 'rb'))
label_encoders = pickle.load(open(label_encoders_path, 'rb'))

destinations_df = pd.read_csv(destinations_path)
userhistory_df = pd.read_csv(userhistory_path)
df = pd.read_csv(df_path)

# Collaborative Filtering Function
# Create a user-item matrix based on user history
user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating')

# Fill missing values with 0 (indicating no rating/experience)
user_item_matrix.fillna(0, inplace=True)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# Function to recommend destinations based on user similarity
def collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df):
    """
    Recommends destinations based on collaborative filtering.

    Args:
    - user_id: ID of the user for whom recommendations are to be made.
    - user_similarity: Cosine similarity matrix for users.
    - user_item_matrix: User-item interaction matrix (e.g., ratings or preferences).
    - destinations_df: DataFrame containing destination details.

    Returns:
    - DataFrame with recommended destinations and their details.
    """
    # Find similar users
    similar_users = user_similarity[user_id - 1]

    # Get the top 5 most similar users
    similar_users_idx = np.argsort(similar_users)[::-1][1:6]

    # Get the destinations liked by similar users
    similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)

    # Recommend the top 5 destinations
    recommended_destinations_ids = similar_user_ratings.sort_values(ascending=False).head(5).index

    # Filter the destinations DataFrame to include detailed information
    recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_destinations_ids)][[
        'DestinationID', 'Name', 'State', 'Type', 'BestTimeToVisit'
    ]]

    return recommendations

# Prediction system
def recommend_destinations(user_input, model, label_encoders, features, data):
    # Encode user input
    encoded_input = {}
    for feature in features:
        if feature in label_encoders:
            encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
        else:
            encoded_input[feature] = user_input[feature]

    # Convert to DataFrame
    input_df = pd.DataFrame([encoded_input])

    # Predict popularity
    predicted_popularity = model.predict(input_df)[0]

    return predicted_popularity

# Helper function to get user_id from email
def get_user_id_from_email(email):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT user_id FROM user_mapping WHERE email = ?', (email,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return result['user_id']
    
    # If not found, check if there's an existing mapping in user_mapping table
    # If not, create a new mapping with the next available user_id
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT MAX(user_id) as max_id FROM user_mapping')
    max_id = cursor.fetchone()['max_id']
    
    if max_id is None:
        next_id = 1
    else:
        next_id = max_id + 1
    
    cursor.execute('INSERT INTO user_mapping (email, user_id) VALUES (?, ?)', (email, next_id))
    conn.commit()
    conn.close()
    
    return next_id

# Route for the Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Login routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['email'] = user['email']
            session['fullname'] = user['fullname']
            return redirect(url_for('recommendation'))
        else:
            return render_template('login.html', error='Invalid email or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
        
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('INSERT INTO users (fullname, email, password) VALUES (?, ?, ?)',
                          (fullname, email, hashed_password))
            conn.commit()
            
            # Also create a mapping for recommendations
            get_user_id_from_email(email)
            
            conn.close()
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('register.html', error='Email already exists')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Route for Travel Recommendation Page
@app.route('/recommendation')
def recommendation():
    if 'email' not in session:
        return redirect(url_for('login'))
    return render_template('recommendation.html')

# Route for the recommendation
@app.route("/recommend", methods=['GET', 'POST'])
def recommend():
    if 'email' not in session:
        return redirect(url_for('login'))
        
    if request.method == "POST":
        # Get user_id from the email in the session
        user_id = get_user_id_from_email(session['email'])
        
        # Capture form data
        user_input = {
            'Name_x': request.form['name'],
            'Type': request.form['type'],
            'State': request.form['state'],
            'BestTimeToVisit': request.form['best_time'],
            'Preferences': request.form['preferences'],
            'Gender': request.form['gender'],
            'NumberOfAdults': request.form['adults'],
            'NumberOfChildren': request.form['children'],
        }

        # Collaborative filtering function
        recommended_destinations = collaborative_recommend(user_id, user_similarity,
                                                          user_item_matrix, destinations_df)

        # Prediction function for popularity (if applicable)
        predicted_popularity = recommend_destinations(user_input, model, label_encoders, features, df)

        # Render the recommendation page with recommendations
        return render_template('recommendation.html', recommended_destinations=recommended_destinations,
                              predicted_popularity=predicted_popularity)
    return render_template('recommendation.html')


if __name__ == '__main__':
    # Print the directory structure for debugging
    print("Current working directory:", os.getcwd())
    print("Templates directory:", os.path.join(app.root_path, app.template_folder))
    print("Does templates directory exist?", os.path.exists(os.path.join(app.root_path, app.template_folder)))
    
    # Check if template files exist
    template_files = ['index.html', 'login.html', 'register.html', 'recommendation.html']
    for template in template_files:
        template_path = os.path.join(app.root_path, app.template_folder, template)
        print(f"Does {template} exist?", os.path.exists(template_path))
    
    # Run the app in debug mode
    app.run(debug=True)