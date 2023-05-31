import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV

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

# Fit the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X_encoded, y)


# Get feature importances
feat_importances = model.feature_importances_
feature_importances_df = pd.DataFrame({'Feature': features, 'Importance': feat_importances})
feature_importances_df = feature_importances_df.sort_values('Importance', ascending=False)

# Select top 8 features
top_features = feature_importances_df.head(8)['Feature'].tolist()
X_selected = pd.DataFrame(X_encoded, columns=features)[top_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train,y_train)

# Get the best estimator and its corresponding parameters
best_estimator = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Parameters:")
print(best_params)

# Fit the RandomForestClassifier model with the best parameters
model = RandomForestClassifier(**best_params)
model.fit(X_train[top_features], y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
# Generate the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Compute the FPR, TPR, and thresholds for the ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(label_binarize(y_test, classes=[1, 2, 3, 4])[:, i], y_pred_proba[:, i])
    roc_auc[i] = roc_auc_score(label_binarize(y_test, classes=[1, 2, 3, 4])[:, i], y_pred_proba[:, i])

# Plot the ROC Curve
plt.figure(figsize=(8, 6))
colors = ['#FF8181', '#FFDDA1', '#C5E99B', '#6AC3C9']  # Adjust colors for each class
class_labels = ['Worst', 'Bad', 'Good', 'Best']
for i in range(4):
    plt.plot(fpr[i], tpr[i], label='ROC Curve (AUC = {:.2f}) for Class {}'.format(roc_auc[i], class_labels[i]),color=colors[i])
    print('ROC Curve (AUC = {:.2f}) for Class {}'.format(roc_auc[i], class_labels[i]))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, cmap='Greens', fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Radar Chart
metrics = ['precision', 'recall', 'f1-score']
values = np.array([report.split()[-4:-1] for report in report.split('\n')[2:-5]], dtype=np.float)
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
for i in range(len(class_labels)):
    values_i = values[i].tolist()
    values_i.append(values_i[0])
    ax.plot(angles, values_i, linewidth=2, linestyle='solid', label=class_labels[i], color=colors[i])
    ax.fill(angles, values_i, alpha=0.25)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=8)
plt.ylim(0, 1)
plt.xticks(angles[:-1], metrics, color="grey", size=10)
plt.title('Classification Metrics by Scores', size=12)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()

# Score distribution Truth VS Predicted
# Get the truth class counts
gt_class_counts = y_test.value_counts()

# Get the predicted class counts
predicted_class_counts = pd.Series(y_pred).value_counts()

# Set up the subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Ground Truth Class Distribution
axes[0].pie(gt_class_counts, labels=class_labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[0].set_title('True Score Distribution')

# Predicted Class Distribution
axes[1].pie(predicted_class_counts, labels=class_labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Predicted Score Distribution')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()
