import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE

# Load the data
data = pd.read_excel('Meal Analysis (2017) .xlsx')

# Remove the "Unnamed: 0" column
data = data.drop("Unnamed: 0", axis=1)

# Data Cleaning
data = data.dropna()  # Remove rows with missing values

X = data.drop('Score(1:worst 2:bad 3:good 4:best)', axis=1)
y = data['Score(1:worst 2:bad 3:good 4:best)']

# Feature Engineering

X['BMI'] = X['weight'] / ((X['height'] / 100) ** 2)

X['Protein_Ratio'] = X['P[g]'] / X['P target(15%)[g]']

X['Fat_Ratio'] = X['F[g]'] / X['F target(25%)[g]']

X['Carbohydrate_Ratio'] = X['C[g]'] / X['C target(60%)[g]']

X['Energy_Ratio'] = X['E[kcal]'] / X['EER[kcal]']

gender_mapping = {'male': True, 'female': False}
X['Sodium_Ratio'] = X['Salt[g]'] / X['gender'].map(gender_mapping).replace({True: 8, False: 7})

X['Vegetable_Ratio'] = X['Vegetables[g]'] / 350

# Data Preprocessing
numeric_features = ['age', 'height', 'weight', 'EER[kcal]', 'P target(15%)[g]', 'F target(25%)[g]',
                    'C target(60%)[g]', 'number of dishes', 'E[kcal]', 'P[g]', 'F[g]', 'C[g]', 'Salt[g]',
                    'Vegetables[g]', 'Vegetable_Ratio', 'Sodium_Ratio', 'Energy_Ratio', 'Carbohydrate_Ratio',
                    'Fat_Ratio', 'Protein_Ratio', 'BMI']
categorical_features = ['Type', 'gender']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_encoded = preprocessor.fit_transform(X)
onehot_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
features = numeric_features + onehot_features

# Define the models
models = [
    RandomForestClassifier(),
    ExtraTreesClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    SVC(),
    KNeighborsClassifier(),
    GaussianNB(),
    MLPClassifier(max_iter=2000)
]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)

# Create a DataFrame to store the evaluation metrics
evaluation_metrics = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])

# For each model
for model in models:
    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Get the classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)

    # Add the evaluation metrics to the DataFrame
    evaluation_metrics = pd.concat([evaluation_metrics, pd.DataFrame({
        'Model': [model.__class__.__name__],
        'Accuracy': [report['accuracy']],
        'Precision': [report['macro avg']['precision']],
        'Recall': [report['macro avg']['recall']],
        'F1_Score': [report['macro avg']['f1-score']]
    })], ignore_index=True)

    # Print the confusion matrix
    print(confusion_matrix(y_test, y_pred))
    print(evaluation_metrics)

# Plot the comparison of models
plt.figure(figsize=(10, 6))
colors = sns.color_palette('viridis', len(evaluation_metrics))
sns.barplot(x='Accuracy', y='Model', data=evaluation_metrics, palette=colors)
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.title('Model Comparison')
plt.show()

# Visualize t-SNE representation of the data
tsne = TSNE(n_components=2, random_state=1)
X_tsne = tsne.fit_transform(X_encoded)

plt.figure(figsize=(8, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='viridis')
plt.title('t-SNE Visualization of Data')
plt.show()

# Additional Comparison Metrics
print("Additional Comparison Metrics:")
print(evaluation_metrics[['Model', 'Precision', 'Recall', 'F1_Score']])


