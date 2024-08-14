# -*- coding: utf-8 -*-
"""Notebook.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12G2-6a67934y9Iy6zOXLVVVFoe_IRQYa
"""

import pandas as pd



ds = pd.read_csv('fitness analysis.csv')

ds

ds.info()

ds.head()

"""# PREPROCESSING"""

new_cols=['Timestamp','Name','Gender','Age','Exercise_importance','Fitness_level','Regularity','Barriers','Exercises','Do_you','Time','Time_spent','Balanced_diet','prevents_balanced','Health_level','Recommend_fitness','Equipment','Motivation']

ds.columns

column_reference=pd.DataFrame(new_cols,ds.columns)
column_reference

ds.columns=new_cols

ds.head()

ds['Age'].unique()

df = ds[(ds['Age'] == '19 to 25')]

df.head()

df = df.reset_index(drop=True)

df.head()

df

df.drop(columns=['Timestamp','Name'],inplace=True)

df['Exercise_importance'].unique()

df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1})

fitness_mapping = {
    'Unfit': 1,
    'Average': 2,
    'Good': 3,
    'Very good': 4,
    'Perfect': 5
}

# Replace fitness levels with numerical values in the 'Fitness_level' column
df['Fitness_level'] = df['Fitness_level'].replace(fitness_mapping)

# Mapping exercise frequency to numerical values
exercise_mapping = {
    'Never': 0,
    '1 to 2 times a week': 1.5,
    '2 to 3 times a week': 2.5,
    '3 to 4 times a week': 3.5,
    '5 to 6 times a week': 5.5,
    'Everyday': 7
}

# Replace exercise frequency with numerical values in the 'Regularity' column
df['Regularity'] = df['Regularity'].replace(exercise_mapping)

df['Time_spent'].unique()

# Mapping exercise durations to numerical values
exercise_duration_mapping = {
    "I don't really exercise": 0,
    '30 minutes': 0.5,
    '1 hour': 1,
    '2 hours': 2,
    '3 hours and above': 3
}

# Replace exercise durations with numerical values in the 'Time_spent' column
df['Time_spent'] = df['Time_spent'].replace(exercise_duration_mapping)

df['Health_level'].unique()

df['Motivation'].unique()

df['Time'].unique()

exercise_time_mapping = {
    "Early morning": 3,
    'Evening': 2,
    'Afternoon': 1,
}

df['Time'] = df['Time'].replace(exercise_time_mapping)

df['Balanced_diet'].unique()

df['Balanced_diet'] = df['Balanced_diet'].replace({'No': 0, 'Not always': 1,'Yes':2})

df['Equipment'].unique()

df['Equipment'] = df['Equipment'].replace({'No': 0,'Yes':1})

df.head()

df.info()

df.to_csv('output_file.csv', index=False)

"""# DESCRIPTIVE STATISTICS"""

df.describe()

df.corr()

df['Exercise_importance'].value_counts()

df['Fitness_level'].value_counts()

df['Time_spent'].value_counts()

df['Health_level'].value_counts()

df['Regularity'].value_counts()

df['Barriers'].value_counts()

df['Time_spent'].value_counts()

df['Health_level'].value_counts()

df['Motivation'].value_counts()

"""# VISUALIZATION"""

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
    sns.histplot(df['Exercise_importance'], kde=True)  # Use histplot instead of distplot
    plt.title('Distribution of Exercise Importance')
    plt.xlabel('Importance of Exercise')
    plt.ylabel('Density')
    plt.show()

exercises_list={}
for selected_options in df['Exercises']:
    for exercise in selected_options.split(";"):
        if exercise in exercises_list:
            exercises_list[exercise]+=1
        else:
            exercises_list[exercise]=1

sorted_list={}
for i in sorted(exercises_list,key=exercises_list.get,reverse=True):
    sorted_list[i]=exercises_list[i]

count=sum(sorted_list.values())
for i in sorted_list:
    sorted_list[i]=(sorted_list[i]/count)*100

sorted_list

plt.bar(sorted_list.keys(),sorted_list.values())
plt.xticks(rotation=90)
plt.title("Exercise preferred by Participants")
plt.ylabel("Percentage")
plt.ylim(0,100)
plt.show()

times=df["Time"].value_counts(normalize=True)*100
plt.pie(times,labels=times.index,explode=(0.05,0.05,0.1),shadow=True,autopct='%.1f%%',startangle=90)
plt.title("Preferred Time to exercise")
plt.show()

motivation_list={}
for selected_options in df['Motivation']:
    for motivation in selected_options.split(";"):
        if motivation in motivation_list:
            motivation_list[motivation]+=1
        else:
            motivation_list[motivation]=1

top_5_motivation=pd.DataFrame.from_dict(motivation_list.items()).sort_values(by=1,ascending=False)[:5]

top_5_motivation[0]=top_5_motivation[0].apply(lambda x:x.replace("I want to ",""))

sns.barplot(x=0,y=1,data=top_5_motivation)
plt.xticks(rotation=45)
plt.ylabel("Number of responses")
plt.title("Top five reasons to exercise")
plt.xlabel("")
plt.show()

barrier_list={}
for selected_options in df['Barriers']:
    for barrier in selected_options.split(";"):
        if barrier in barrier_list:
            barrier_list[barrier]+=1
        else:
            barrier_list[barrier]=1

top_5_barrier=pd.DataFrame.from_dict(barrier_list.items()).sort_values(by=1,ascending=False)[:6].drop(3,axis=0)

top_5_barrier["Percentage"]=(top_5_barrier[1]/top_5_barrier[1].sum())*100

top_5_barrier

plt.pie(top_5_barrier["Percentage"],labels=top_5_barrier[0],autopct="%.1f%%",shadow=True)
centre_circle = plt.Circle((0,0),0.8,color='yellow', fc='white',linewidth=1.5)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("Top 5 barriers to daily exercise")
plt.show()

df.info()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Fitness_level', y='Exercise_importance', data=df, palette='muted')
plt.title('Box Plot: Exercise Importance by Fitness Level')
plt.xlabel('Fitness Level')
plt.ylabel('Exercise Importance')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Health_level', y='Exercise_importance', data=df, palette='pastel')
plt.title('Bar Plot: Average Exercise Importance by Health Level')
plt.xlabel('Health Level')
plt.ylabel('Average Exercise Importance')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='Health_level', y='Exercise_importance', hue='Fitness_level', data=df, palette='Set2')
plt.title('Grouped Bar Plot: Exercise Importance by Health and Fitness Level')
plt.xlabel('Health Level')
plt.ylabel('Average Exercise Importance')
plt.legend(title='Fitness Level')
plt.show()

df.hist(figsize=(12, 8), bins=20)
plt.show()

df.head()

"""# CLASSIFICATION"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define features (X) and target variable (y)
X = df[['Exercise_importance', 'Regularity', 'Time', 'Time_spent', 'Balanced_diet', 'Health_level', 'Equipment']]
y = df['Fitness_level']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, zero_division=1)

    # Print results
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", classification_rep)
    print("="*50)

"""# HYPERPARAMETERS TUNING"""

from sklearn.model_selection import GridSearchCV

# Decision Tree hyperparameter tuning
param_grid = {'max_depth': [3, 5, 7, None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

dt_model = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(dt_model, param_grid, cv=5)
dt_grid.fit(X_train, y_train)

# Best hyperparameters
best_params_dt = dt_grid.best_params_
print("Best Hyperparameters (Decision Tree):", best_params_dt)

# Evaluate the tuned model
y_pred_dt = dt_grid.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
classification_rep_dt = classification_report(y_test, y_pred_dt, zero_division=1)
print("\nDecision Tree Results:")
print(f"Accuracy: {accuracy_dt}")
print("Classification Report:\n", classification_rep_dt)

# Random Forest hyperparameter tuning
param_grid_rf = {'n_estimators': [50, 100, 150],
                 'max_depth': [None, 10, 20, 30],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 4]}

rf_model = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_model, param_grid_rf, cv=5)
rf_grid.fit(X_train, y_train)

# Best hyperparameters
best_params_rf = rf_grid.best_params_
print("Best Hyperparameters (Random Forest):", best_params_rf)

# Evaluate the tuned model
y_pred_rf = rf_grid.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf, zero_division=1)
print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_rf}")
print("Classification Report:\n", classification_rep_rf)

# Feature importance in Random Forest
feature_importance_rf = rf_grid.best_estimator_.feature_importances_
print("Feature Importance (Random Forest):", feature_importance_rf)

# Gradient Boosting hyperparameter tuning
param_grid_gb = {'n_estimators': [50, 100, 150],
                 'learning_rate': [0.01, 0.1, 0.2],
                 'max_depth': [3, 5, 7]}

gb_model = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(gb_model, param_grid_gb, cv=5)
gb_grid.fit(X_train, y_train)

# Best hyperparameters
best_params_gb = gb_grid.best_params_
print("Best Hyperparameters (Gradient Boosting):", best_params_gb)

# Evaluate the tuned model
y_pred_gb = gb_grid.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
classification_rep_gb = classification_report(y_test, y_pred_gb, zero_division=1)
print("\nGradient Boosting Results:")
print(f"Accuracy: {accuracy_gb}")
print("Classification Report:\n", classification_rep_gb)

# Feature importance in Gradient Boosting
feature_importance_gb = gb_grid.best_estimator_.feature_importances_
print("Feature Importance (Gradient Boosting):", feature_importance_gb)

# Logistic Regression hyperparameter tuning
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                 'penalty': ['l2']}

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_grid = GridSearchCV(lr_model, param_grid_lr, cv=5)
lr_grid.fit(X_train, y_train)

# Best hyperparameters
best_params_lr = lr_grid.best_params_
print("Best Hyperparameters (Logistic Regression):", best_params_lr)

# Evaluate the tuned model
y_pred_lr = lr_grid.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
classification_rep_lr = classification_report(y_test, y_pred_lr, zero_division=1)
print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_lr}")
print("Classification Report:\n", classification_rep_lr)

# SVM hyperparameter tuning
param_grid_svm = {'C': [0.1, 1, 10, 100],
                  'kernel': ['linear', 'rbf', 'poly'],
                  'degree': [2, 3, 4],
                  'gamma': ['scale', 'auto']}

svm_model = SVC(random_state=42)
svm_grid = GridSearchCV(svm_model, param_grid_svm, cv=5)
svm_grid.fit(X_train, y_train)

# Best hyperparameters
best_params_svm = svm_grid.best_params_
print("Best Hyperparameters (SVM):", best_params_svm)

# Evaluate the tuned model
y_pred_svm = svm_grid.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
classification_rep_svm = classification_report(y_test, y_pred_svm, zero_division=1)
print("\nSupport Vector Machine Results:")
print(f"Accuracy: {accuracy_svm}")
print("Classification Report:\n", classification_rep_svm)

# k-NN hyperparameter tuning
param_grid_knn = {'n_neighbors': [3, 5, 7, 9],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

knn_model = KNeighborsClassifier()
knn_grid = GridSearchCV(knn_model, param_grid_knn, cv=5)
knn_grid.fit(X_train, y_train)

# Best hyperparameters
best_params_knn = knn_grid.best_params_
print("Best Hyperparameters (k-NN):", best_params_knn)

# Evaluate the tuned model
y_pred_knn = knn_grid.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
classification_rep_knn = classification_report(y_test, y_pred_knn, zero_division=1)
print("\nk-Nearest Neighbors Results:")
print(f"Accuracy: {accuracy_knn}")
print("Classification Report:\n", classification_rep_knn)

"""# Handling Imbalanced Classes"""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

# Example with Random Forest and SMOTE
rf_model_imb = RandomForestClassifier(random_state=42)
smote = SMOTE(random_state=42)

# Create a pipeline with SMOTE
pipeline = make_pipeline(smote, rf_model_imb)

# Train and evaluate the model with imbalanced classes
pipeline.fit(X_train, y_train)
y_pred_rf_imb = pipeline.predict(X_test)

# Evaluate the model with imbalanced classes
accuracy_rf_imb = accuracy_score(y_test, y_pred_rf_imb)
classification_rep_rf_imb = classification_report(y_test, y_pred_rf_imb, zero_division=1)
print("\nRandom Forest Results with SMOTE (Handling Imbalanced Classes):")
print(f"Accuracy: {accuracy_rf_imb}")
print("Classification Report:\n", classification_rep_rf_imb)

"""# Cross Validation"""

from sklearn.model_selection import cross_val_score

# Example with Decision Tree and cross-validation
dt_model_cv = DecisionTreeClassifier(random_state=42)

# Perform cross-validation
cv_scores_dt = cross_val_score(dt_model_cv, X_train, y_train, cv=5)

# Print cross-validation scores
print("Cross-Validation Scores (Decision Tree):", cv_scores_dt)
print("Mean CV Accuracy (Decision Tree):", cv_scores_dt.mean())

from sklearn.model_selection import cross_val_score

# Example with Random Forest and cross-validation
rf_model_cv = RandomForestClassifier(random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf_model_cv, X_train, y_train, cv=5)

# Print cross-validation scores
print("Cross-Validation Scores (Random Forest):", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

"""# Prediction"""

# Define features (X) and target variable (y)
X = df[['Exercise_importance', 'Regularity', 'Time', 'Time_spent', 'Balanced_diet', 'Health_level', 'Equipment']]
y = df['Fitness_level']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Create a new data point within the specified constraints
new_data_point = [3, 13.5, 2, 3.5, 1, 3, 0]  # Example data point

# Convert the new data point to a DataFrame
new_data_df = pd.DataFrame([new_data_point], columns=['Exercise_importance', 'Regularity', 'Time', 'Time_spent', 'Balanced_diet', 'Health_level', 'Equipment'])

# Make predictions using the Decision Tree model
prediction_dt = dt_model.predict(new_data_df)

# Make predictions using the Random Forest model
prediction_rf = rf_model.predict(new_data_df)

# Map predicted fitness levels to descriptions
fitness_level_descriptions = {
    1: "Unfit",
    2: "Average",
    3: "Good",
    4: "Very Good",
    5: "Perfect"
}

# Display the input values
print("\nInput Data:")
print("Exercise Importance:", new_data_point[0])
print("Regularity:", new_data_point[1])
print("Time:", new_data_point[2])
print("Time Spent:", new_data_point[3])
print("Balanced Diet:", new_data_point[4])
print("Health Level:", new_data_point[5])
print("Equipment:", new_data_point[6])

# Display the predicted fitness level by Decision Tree
print("\nPredicted Fitness Level By Decision Tree:", fitness_level_descriptions[prediction_dt[0]])

# Display the predicted fitness level by Random Forest
print("Predicted Fitness Level By Random Forest:", fitness_level_descriptions[prediction_rf[0]])

# Create new data points within the specified constraints
new_data_points = [
    [3, 13.5, 2, 3.5, 1, 3, 0],  # Example data point 1
    [4, 12.0, 3, 2.0, 2, 4, 1],  # Example data point 2
]

# Convert the new data points to a DataFrame
new_data_df = pd.DataFrame(new_data_points, columns=['Exercise_importance', 'Regularity', 'Time', 'Time_spent', 'Balanced_diet', 'Health_level', 'Equipment'])

# Make predictions using the Decision Tree model
predictions_dt = dt_model.predict(new_data_df)

# Make predictions using the Random Forest model
predictions_rf = rf_model.predict(new_data_df)

# Map predicted fitness levels to descriptions
fitness_level_descriptions = {
    1: "Unfit",
    2: "Average",
    3: "Good",
    4: "Very Good",
    5: "Perfect"
}

# Display the input values and predicted fitness levels
for i, new_data_point in enumerate(new_data_points):
    print(f"\nInput Data Point {i + 1}:")
    print("Exercise Importance:", new_data_point[0])
    print("Regularity:", new_data_point[1])
    print("Time:", new_data_point[2])
    print("Time Spent:", new_data_point[3])
    print("Balanced Diet:", new_data_point[4])
    print("Health Level:", new_data_point[5])
    print("Equipment:", new_data_point[6])

    # Display the predicted fitness level by Decision Tree
    print("\nPredicted Fitness Level By Decision Tree:", fitness_level_descriptions[predictions_dt[i]])

    # Display the predicted fitness level by Random Forest
    print("Predicted Fitness Level By Random Forest:", fitness_level_descriptions[predictions_rf[i]])

new_data_points = [
    [3, 12.5, 2, 3.5, 1, 3, 0],  # Example 1
    [4, 9.2, 1, 2.0, 0, 4, 1],  # Example 2
    [2, 6.7, 0, 1.5, 2, 2, 0],  # Example 3
    [5, 4.8, 2, 4.0, 1, 5, 1],  # Example 4
    [3, 8.5, 1, 2.5, 0, 3, 1]   # Example 5
]

# Convert the new data points to a DataFrame
new_data_df = pd.DataFrame(new_data_points, columns=['Exercise_importance', 'Regularity', 'Time', 'Time_spent', 'Balanced_diet', 'Health_level', 'Equipment'])

# Make predictions using the Decision Tree model
predictions_dt = dt_model.predict(new_data_df)

# Make predictions using the Random Forest model
predictions_rf = rf_model.predict(new_data_df)

# Map predicted fitness levels to descriptions
fitness_level_descriptions = {
    1: "Unfit",
    2: "Average",
    3: "Good",
    4: "Very Good",
    5: "Perfect"
}

# Display the input values and predicted fitness levels
for i, new_data_point in enumerate(new_data_points):
    print(f"\nInput Data Point {i + 1}:")
    print("Exercise Importance:", new_data_point[0])
    print("Regularity:", new_data_point[1])
    print("Time:", new_data_point[2])
    print("Time Spent:", new_data_point[3])
    print("Balanced Diet:", new_data_point[4])
    print("Health Level:", new_data_point[5])
    print("Equipment:", new_data_point[6])

    # Display the predicted fitness level by Decision Tree
    print("\nPredicted Fitness Level By Decision Tree:", fitness_level_descriptions[predictions_dt[i]])

    # Display the predicted fitness level by Random Forest
    print("Predicted Fitness Level By Random Forest:", fitness_level_descriptions[predictions_rf[i]])

