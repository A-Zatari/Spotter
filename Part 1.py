import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

def process_dates(df):
    """Extracts numerical time-series features from a date string."""
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['day'] = df_copy['date'].dt.day
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek
    return df_copy

def predict_validation_set(df_train, val_filepath, output_filename):
    """Evaluates 3 models, retrains the best on all data, and predicts validation inputs."""
    print("--- Running Validation Pipeline ---")
    features = ['weight', 'distance', 'quote_signal']
    target = 'posted_rate'

    X_full = df_train[features]
    y_full = df_train[target]
    
    # Split temporarily to evaluate and find the best model
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    preprocessor = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=42, max_iter=10)),
        ('scaler', StandardScaler())
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42)
    }

    best_accuracy = -np.inf
    best_pipeline = None
    best_name = ""

    print("Evaluating models to find the best fit...")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        accuracy = (1 - mape) * 100
        
        print(f"  {name} -> Accuracy: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_pipeline = pipeline
            best_name = name

    print(f">> Selected Best Model: {best_name} (Accuracy: {best_accuracy:.2f}%)")

    # Retrain the winning model architecture on 100% of the training data
    print(f"Retraining {best_name} on the entire dataset...")
    best_pipeline.fit(X_full, y_full)

    # Load validation data relative to the script directory, keeping the first column as index
    df_val = pd.read_csv(val_filepath, index_col=0)
    X_val = df_val[features]

    # Predict using the newly retrained pipeline
    df_val['predicted_rate'] = best_pipeline.predict(X_val)
    
    # Export only the prediction column alongside the preserved index
    df_final = df_val[['predicted_rate']]
    df_final.to_csv(output_filename, index=True)
    print(f">> Validation predictions saved to {output_filename}\n")


def predict_december_set(df_train, dec_filepath, output_filename):
    """Trains on overlapping categorical/date features to predict December inputs."""
    print("--- Running December Pipeline ---")
    
    df_train_processed = process_dates(df_train)
    
    features_num = ['distance', 'weight', 'month', 'day', 'dayofweek']
    features_cat = ['equipment', 'pickup', 'delivery']
    features = features_num + features_cat
    target = 'posted_rate'

    X_train = df_train_processed[features]
    y_train = df_train_processed[target]

    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=42, max_iter=10)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features_num),
            ('cat', categorical_transformer, features_cat)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=150, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Load and process the target December dataset relative to script directory
    df_dec = pd.read_csv(dec_filepath)
    df_dec_processed = process_dates(df_dec)
    
    X_dec = df_dec_processed[features]

    # Generate predictions and map back to the original unmodified dataframe
    df_dec['predicted_rate'] = pipeline.predict(X_dec)
    
    df_dec.to_csv(output_filename, index=False)
    print(f">> December predictions saved to {output_filename}\n")


def main():
    # Dynamically resolve the directory where this script file lives
    script_dir = pathlib.Path(__file__).parent.resolve()

    # Define paths relative to the script location
    train_file = script_dir / "train-test.csv"
    val_file = script_dir / "data" / "validation.csv"          # Adjust folder name if needed
    dec_file = script_dir / "data" / "december-chart-inputs.csv" # Adjust folder name if needed
    
    val_output = script_dir / "validation_with_predictions.csv"
    dec_output = script_dir / "december-chart-predictions.csv"

    # Fallback if files are directly in the same folder as the script
    if not val_file.exists() and (script_dir / "validation.csv").exists():
        val_file = script_dir / "validation.csv"
    if not dec_file.exists() and (script_dir / "december-chart-inputs.csv").exists():
        dec_file = script_dir / "december-chart-inputs.csv"

    # Load the main training set into memory once
    print(f"Loading training dataset from: {train_file}\n")
    df_train = pd.read_csv(train_file)

    # Execute both workflows
    predict_validation_set(df_train, val_file, val_output)
    predict_december_set(df_train, dec_file, dec_output)


if __name__ == "__main__":
    main()