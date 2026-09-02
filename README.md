# ✈️ TripGenie

### 🌍 AI-Powered Travel Recommendation System

TripGenie is an AI-powered travel recommendation system designed to help users discover destinations based on their travel preferences and interests.

The application combines Machine Learning, data processing, and web development to provide personalized travel recommendations through an easy-to-use Flask web application.

Instead of manually searching through numerous destinations, users can provide their preferences and TripGenie analyzes the available destination data to recommend places that best match their interests.

---

## ✨ Features

- 🤖 Personalized travel recommendations
- 📐 Cosine similarity-based recommendation system
- 🔐 User registration and login
- 🔑 Secure password hashing
- 👤 Session-based user authentication
- 🗄️ SQLite database integration
- 📊 Data processing and visualization
- 🌐 Flask-based web application
- 💾 Pre-trained model and label encoder support
- 📱 Simple and user-friendly interface

---

## 🧠 How TripGenie Works

TripGenie follows a content-based recommendation approach.

The system processes destination and user-related information and represents relevant features as numerical vectors.

Cosine similarity is then used to measure the similarity between the user's preferences and available destinations.

The destinations with higher similarity scores are ranked and presented as recommendations.

### Recommendation Flow

```text
        👤 User
          │
          ▼
   🔐 Register / Login
          │
          ▼
 🎯 Enter Travel Preferences
          │
          ▼
   📊 Process User Data
          │
          ▼
  🔢 Feature Representation
          │
          ▼
   📐 Cosine Similarity
          │
          ▼
   📋 Rank Destinations
          │
          ▼
 ✈️ Personalized Recommendations

TripGenie/
│
├── 📁 code_and_dataset/
│   ├── dataset.py
│   ├── Expanded_Destinations.csv
│   ├── Final_dlt.csv
│   ├── Final_Updated_Expanded_Reviews.csv
│   ├── Final_Updated_Expanded_UserHistory.csv
│   └── Final_Updated_Expanded_Users.csv
│
├── 📁 Static_videos/
│   └── Travel Agency Logo Advert.mp4
│
├── 📁 templates/
│   ├── index.html
│   ├── login.html
│   ├── recommendation.html
│   └── register.html
│
├── 📄 app.py
├── 📄 main.py
├── 📄 visualization.py
│
├── 🤖 model.pkl
├── 🏷️ label_encoder.pkl
├── 🗄️ users.db
│
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 README.md


User Preferences
       │
       ▼
Feature Representation
       │
       ▼
Destination Feature Vectors
       │
       ▼
Cosine Similarity
       │
       ▼
Similarity Scores
       │
       ▼
Ranked Destinations
       │
       ▼
Recommended Destinations


➡️Follow the steps below to run TripGenie locally.

1. Clone the Repository
git clone https://github.com/YOUR-USERNAME/TripGenie.git

Navigate into the project directory:

cd TripGenie
2. Create a Virtual Environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate

For macOS/Linux:

python3 -m venv venv
source venv/bin/activate
3. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt

If you haven't created requirements.txt yet, you can generate it from your current environment using:

pip freeze > requirements.txt
4. Run the Application

Start the Flask application:

python app.py

The application should start on the local Flask development server.

Open the local address shown in your terminal, typically:

http://127.0.0.1:5000/


TripGenie can be further enhanced with:

🧠 Advanced recommendation algorithms
🤖 AI-powered conversational travel assistant
💰 Budget-based trip recommendations
🌦️ Real-time weather integration
🏨 Hotel recommendations
✈️ Flight information integration
🗺️ Interactive maps
📍 Location-aware recommendations
⭐ User ratings and feedback
📱 Responsive mobile-first interface
☁️ Cloud deployment
🔄 Real-time travel data integration
