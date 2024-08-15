from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

df = pd.read_csv("output_file.csv")

# Define features (X) and target variable (y)
X = df[['Exercise_importance', 'Regularity', 'Time', 'Time_spent', 'Balanced_diet', 'Health_level', 'Equipment']]
y = df['Fitness_level']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained Decision Tree model and features
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
features = ['Exercise_importance', 'Regularity', 'Time', 'Time_spent', 'Balanced_diet', 'Health_level', 'Equipment']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        user_input = [request.form[feature] for feature in features]

        # Convert the input to the format expected by the model
        user_input = [float(value) if feature == 'Regularity' or feature == 'Time_spent' else int(value) for feature, value in zip(features, user_input)]

        # Make a prediction
        prediction = rf_clf.predict([user_input])[0]

        # Map predicted fitness level to description
        fitness_level_description = {
            1: "Unfit",
            2: "Average",
            3: "Good",
            4: "Very Good",
            5: "Perfect"
        }

        # Display the prediction in the template
        return render_template('result.html', prediction=fitness_level_description[prediction])

if __name__ == '__main__':
    app.run(debug=True)
