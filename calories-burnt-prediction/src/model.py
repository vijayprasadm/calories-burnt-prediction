import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Preprocess data
def preprocess_data(data):
    # Convert categorical variable to dummy/indicator variables
    data = pd.get_dummies(data, columns=['activity'], drop_first=True)
    return data

# Train model
def train_model(data):
    X = data.drop('calories_burned', axis=1)
    y = data['calories_burned']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print(f'Mean Squared Error: {mse}')

    return model

# Visualize predictions
def plot_predictions(y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Calories Burned')
    plt.ylabel('Predicted Calories Burned')
    plt.title('Actual vs Predicted Calories Burned')
    plt.show()

if __name__ == "__main__":
    # File path to dataset
    data = load_data('data/calories_data.csv')
    processed_data = preprocess_data(data)
    model = train_model(processed_data)

    # For visualization
    X_test = processed_data.drop('calories_burned', axis=1).iloc[-20:]  # Last 20 records for prediction
    y_test = processed_data['calories_burned'].iloc[-20:]  # Last 20 actual values
    predictions = model.predict(X_test)
    plot_predictions(y_test, predictions)
