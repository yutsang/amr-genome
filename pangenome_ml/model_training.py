import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import shap
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate_models(df, targets):
    features = df.columns.difference(targets).difference(['ID'])
    
    label_encoders = {}
    
    for target in targets:
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])
        label_encoders[target] = le
        
        print(f"\nEvaluating model for target: {target}")
        
        train_and_evaluate(df, features, target)

def train_and_evaluate(df, features, target):
    
    X = df[features]
    y = df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify categorical features for encoding...
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Create a preprocessor for handling categorical variables...
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'  # Keep other columns as they are.
    )

    # Initialize models with preprocessing in a pipeline...
    models = {
        'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier())]),
        'Random Forest': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())]),
        'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier())]),
        'SVM': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC(gamma='scale', probability=True))]),  # Set gamma explicitly
        'XGBoost': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))])
    }

    # Train each model and evaluate performance...
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.2f}")

        # Use SHAP for feature importance interpretation...
        if name == 'SVM':
            explainer = shap.KernelExplainer(model.named_steps['classifier'].predict_proba,
                                              preprocessor.transform(X_train.sample(frac=0.1)))
            shap_values = explainer.shap_values(preprocessor.transform(X_test))
        else:
            explainer = shap.Explainer(model.named_steps['classifier'], preprocessor.transform(X_train))
            shap_values = explainer(preprocessor.transform(X_test))

        if len(shap_values.values.shape) == 3:
            shap_values_df = pd.DataFrame(shap_values.values[:, :, 0], columns=X.columns)  
        else:
            shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)

        # Average SHAP values for each feature...
        mean_shap_values = shap_values_df.mean().sort_values(ascending=False)

        # Save average SHAP values to CSV file...
        output_dir = f'./output/{name}'
        os.makedirs(output_dir, exist_ok=True)

        mean_shap_values.to_csv(f'{output_dir}/mean_shap_values_{target}.csv')

        # Plot top 20 features based on average SHAP values...
        plt.figure(figsize=(10, 6))
        mean_shap_values.head(20).plot(kind='barh')
         
        plt.title(f'Top 20 Features by Mean SHAP Values for {target}')
        plt.xlabel('Mean SHAP Value')
        plt.ylabel('Features')
        
        plt.savefig(f'{output_dir}/mean_shap_plot_{target}.png')
