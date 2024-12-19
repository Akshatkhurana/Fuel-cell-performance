# Step 1: Load Libraries and Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
dataset_path = "C:\\Users\\khura\\OneDrive\\Desktop\\MLLabEval\\Fuel_cell_performance_data-Full.csv"
df = pd.read_csv(dataset_path)

# Step 2: Preprocess Dataset
# Display basic info
print("Dataset Info:")
print(df.info())

# Check for null values
if df.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    df = df.dropna()  # You can modify this step based on your requirements

# Step 3: Select Target Based on Roll Number
# Replace 'your_roll_number' with your actual roll number.
roll_number = "102203644"  # Example roll number
last_digit = int(roll_number[-1])

# Map roll number ending to target
target_map = {
    0: "Target1", 5: "Target1",
    1: "Target2", 6: "Target2",
    2: "Target3", 7: "Target3",
    3: "Target4", 8: "Target4",
    4: "Target5", 9: "Target5"
}
selected_target = target_map[last_digit]
print(f"\nSelected Target: {selected_target}")

# Drop other targets
df = df[[selected_target, *df.columns.difference([selected_target])]]

# Step 4: Split Dataset (70/30)
X = df.drop(columns=[selected_target])  # Features
y = df[selected_target]  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Run Multiple Prediction Models
# Helper function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    return predictions

# Linear Regression
lr = LinearRegression()
evaluate_model(lr, X_train, X_test, y_train, y_test)

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
evaluate_model(dt, X_train, X_test, y_train, y_test)

# Random Forest
rf = RandomForestRegressor(random_state=42)
evaluate_model(rf, X_train, X_test, y_train, y_test)

# Step 6: Save Results and Upload to GitHub
results = {
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "MSE": [mean_squared_error(y_test, lr.predict(X_test)),
            mean_squared_error(y_test, dt.predict(X_test)),
            mean_squared_error(y_test, rf.predict(X_test))],
    "R2 Score": [r2_score(y_test, lr.predict(X_test)),
                 r2_score(y_test, dt.predict(X_test)),
                 r2_score(y_test, rf.predict(X_test))]
}

results_df = pd.DataFrame(results)
print("\nFinal Results:")
print(results_df)
