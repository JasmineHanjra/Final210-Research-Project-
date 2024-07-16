import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

try:
    # Load dataset
    print("Loading dataset...")
    spotify_data = pd.read_csv('spotify_tracks.csv')
    print(spotify_data.head())
    print(spotify_data.info())

    # Handle missing values
    print("Handling missing values...")
    spotify_data.dropna(inplace=True)

    # Convert duration_ms to minutes
    print("Converting duration_ms to duration_min...")
    spotify_data['duration_min'] = spotify_data['duration_ms'] / 60000.0

    # Correlation heatmap for numeric features
    print("Plotting correlation heatmap...")
    numeric_columns = ['popularity', 'duration_min']
    correlation_matrix = spotify_data[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # Different Genre Analysis
    print("Analyzing genres...")
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='genre', y='popularity', data=spotify_data)
    plt.title('Genre vs. Popularity')
    plt.xticks(rotation=45)
    plt.show()

    # Explicit Content Analysis
    print("Analyzing explicit content...")
    plt.figure(figsize=(8, 6))
    sns.barplot(x='explicit', y='popularity', data=spotify_data)
    plt.title('Explicit Content vs. Popularity')
    plt.xlabel('Explicit')
    plt.ylabel('Popularity')
    plt.show()

    # Artist-specific Analysis: Top 10 artists by average popularity
    print("Analyzing top artists...")
    top_artists = spotify_data.groupby('artists')['popularity'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    top_artists.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Artists by Average Popularity')
    plt.xlabel('Artist')
    plt.ylabel('Average Popularity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Album Analysis: Top 10 albums by average popularity
    print("Analyzing top albums...")
    top_albums = spotify_data.groupby('album')['popularity'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    top_albums.plot(kind='bar', color='lightgreen')
    plt.title('Top 10 Albums by Average Popularity')
    plt.xlabel('Album')
    plt.ylabel('Average Popularity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Outlier detection and handling (using IQR method)
    print("Handling outliers...")
    Q1 = spotify_data['popularity'].quantile(0.25)
    Q3 = spotify_data['popularity'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    spotify_data = spotify_data[(spotify_data['popularity'] >= lower_bound) & (spotify_data['popularity'] <= upper_bound)]

    # Distribution plots
    print("Plotting distribution of popularity...")
    plt.figure(figsize=(10, 6))
    sns.histplot(spotify_data['popularity'], bins=30, kde=True)
    plt.title('Distribution of Song Popularity')
    plt.show()

    # Pair plot for feature relationships
    print("Plotting pair plot...")
    sns.pairplot(spotify_data[['popularity', 'duration_min']])
    plt.show()

    # Encode categorical variables
    print("Encoding categorical variables...")
    spotify_data_encoded = pd.get_dummies(spotify_data, columns=['genre'], drop_first=True)

    # Features and target variable
    print("Defining features and target variable...")
    features = ['duration_min'] + list(spotify_data_encoded.columns[spotify_data_encoded.columns.str.startswith('genre_')])
    X = spotify_data_encoded[features]
    y = spotify_data_encoded['popularity']

    # Train-test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression Model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Evaluate model performance
    print("Evaluating model performance...")
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error (Linear Regression): {mse}')

    # Predicted vs actual values plot
    print("Plotting predicted vs actual values...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
    plt.title('Linear Regression: Predicted vs Actual Popularity')
    plt.xlabel('Actual Popularity')
    plt.ylabel('Predicted Popularity')
    plt.show()

    # Coefficients
    print("Printing coefficients...")
    print('Coefficients:')
    for feature, coef in zip(features, model.coef_):
        print(f'{feature}: {coef}')

    feature_names = X.columns
    coefficients = model.coef_

    # Plotting feature importances
    print("Plotting feature importances...")
    plt.figure(figsize=(12, 8))
    plt.bar(feature_names, coefficients, color='skyblue')
    plt.xlabel('Feature')
    plt.ylabel('Coefficient')
    plt.title('Feature Importances in Linear Regression Model')
    plt.xticks(rotation=90)
    plt.show()

    # Residual Analysis
    print("Performing residual analysis...")
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='blue')
    plt.title('Residual Analysis')
    plt.xlabel('Predicted Popularity')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.show()

    # Model Comparison
    print("Comparing models...")
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'{name} - MSE: {mse}, MAE: {mae}, R2: {r2}')

   # Define the parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    # Perform GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                            param_grid=param_grid,
                            cv=5,
                            scoring='neg_mean_squared_error',
                            verbose=0,  # Increase verbosity to see progress
                            n_jobs=-1)  # Use all available CPU cores

    # Fit the grid search to find the best model
    grid_search.fit(X_train, y_train)

    # Retrieve the best model found by GridSearchCV
    best_rf = grid_search.best_estimator_

    # Evaluate the best model on the test set
    y_pred_best_rf = best_rf.predict(X_test)
    mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
    print(f'Best Random Forest - MSE: {mse_best_rf}')

    # Print the best parameters found by GridSearchCV
    print(f'Best parameters: {grid_search.best_params_}')

    # plot feature importances for the best Random Forest model
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], color='r', align='center')
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
    plt.show()

    # perform residual analysis for the best Random Forest model
    residuals_best_rf = y_test - y_pred_best_rf
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_best_rf, residuals_best_rf, color='blue')
    plt.title('Residual Analysis (Random Forest)')
    plt.xlabel('Predicted Popularity')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.show()


    # Scaling numerical features for model improvement
    print("Scaling numerical features...")
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled['duration_min'] = scaler.fit_transform(X_scaled[['duration_min']])

    # Training and test sets for scaled data
    print("Splitting scaled data into train and test sets...")
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and fit the model with scaled data
    print("Training Linear Regression model with scaled data...")
    model_scaled = LinearRegression()
    model_scaled.fit(X_train_scaled, y_train_scaled)

    y_pred_scaled = model_scaled.predict(X_test_scaled)
    mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
    print(f'Mean Squared Error (Scaled Linear Regression): {mse_scaled}')

    # Function to predict popularity
    def predict_popularity(duration_min, genre, spotify_data, model):
        spotify_data_encoded = pd.get_dummies(spotify_data, columns=['genre'], drop_first=True)
        genre_columns = [col for col in spotify_data_encoded.columns if col.startswith('genre_')]
        new_sample_data = {'duration_min': [duration_min]}
        for genre_col in genre_columns:
            new_sample_data[genre_col] = [1 if genre_col == f'genre_{genre}' else 0]
        
        new_sample = pd.DataFrame(new_sample_data)
        new_sample['duration_min'] = scaler.transform(new_sample[['duration_min']])
        predicted_popularity = model.predict(new_sample)[0]
        return predicted_popularity

    # Example usage
    print("Predicting popularity for example input...")
    predicted_popularity = predict_popularity(3.5, 'Pop', spotify_data, model_scaled)
    print(f'Predicted Popularity: {predicted_popularity}')

    print("Script executed successfully!")
    
except Exception as e:
    print(f"An error occurred: {e}")
