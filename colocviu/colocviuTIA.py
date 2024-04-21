# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance

# Import the new dataset
from ucimlrepo import fetch_ucirepo

# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 

# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features.copy()
y = statlog_german_credit_data.data.targets.copy()

# Data preprocessing

# Drop duplicates
X.drop_duplicates(inplace=True)
X.reset_index(drop=True, inplace=True)


# Encode categorical variables using label encoding
categorical_features = X.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

# Apply label encoding to each categorical feature
X_encoded_categorical = X[categorical_features].apply(label_encoder.fit_transform)

# Concatenate numerical features and label encoded categorical features
X_encoded = pd.concat([X.select_dtypes(include=['number']), X_encoded_categorical], axis=1)

# Correlation analysis
numeric_features = X_encoded.select_dtypes(include=['number'])
features = list(numeric_features.columns)

fig = px.imshow(
    numeric_features.corr().round(2),
    text_auto=True,
    aspect="auto",
    x=features,
    y=features,
    color_continuous_scale="Viridis",
    labels=dict(color="Corelare"),
    title='Analiza de corelație pentru caracteristici independente',
)
fig.update_layout(
    height=800,
    width=1600,
    xaxis_title="Caracteristici",
    yaxis_title="Caracteristici",
    xaxis_nticks=len(features),
    yaxis_nticks=len(features),
    font=dict(family="Arial", size=12),
)
#fig.write_image('fig.png', engine='kaleido')

# Copy the original encoded data (20 attributes) for SVM on Original Data
X_encoded_original = X_encoded.copy()

# Apply PCA to handle correlation between attribute 2 and 5
correlated_features = ['Attribute2', 'Attribute5']

# PCA with one component
pca = PCA(n_components=1)

# PCA data fitting
pca_result = pca.fit_transform(X_encoded[correlated_features])

# Create a new column with PCA result
X_encoded['Attribute2_5_PCA'] = pca_result.squeeze()

# Drop original correlated features
X_encoded.drop(correlated_features, axis=1, inplace=True)

# New heatmap with dropped features and PCA
features = list(X_encoded.corr().round(2).columns)  # Drop 'class' from the list of features
fig2 = px.imshow(
    X_encoded.corr().round(2),
    text_auto=True,
    aspect="auto",
    x=features,
    y=features,
    color_continuous_scale="Viridis",
    labels=dict(color="Corelare"),
    title='Analiza de corelație pentru caracteristici independente',
)
fig2.update_layout(
    height=800,
    width=1600,
    xaxis_title="Caracteristici",
    yaxis_title="Caracteristici",
    xaxis_nticks=len(features),
    yaxis_nticks=len(features),
    font=dict(family="Arial", size=12),
)
#fig2.write_image('fig2.png', engine='kaleido')

#Select the first column from y
y = y.iloc[:, 0]

# Model preparation, model itself and results
# Split the data into training and testing sets for both original and reduced data
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_encoded_original, y, test_size=0.2, random_state=42)
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Normalization through min-max scaling for both original and reduced data
scaler = MinMaxScaler()
X_train_original_scaled = scaler.fit_transform(X_train_original)
X_test_original_scaled = scaler.transform(X_test_original)
X_train_reduced_scaled = scaler.fit_transform(X_train_reduced)
X_test_reduced_scaled = scaler.transform(X_test_reduced)

# Reference accuracy for Original Data
reference_model_original = DummyClassifier(strategy='most_frequent')
reference_model_original.fit(X_train_original_scaled, y_train_original)
y_pred_reference_original = reference_model_original.predict(X_test_original_scaled)
accuracy_reference_original = accuracy_score(y_test_original, y_pred_reference_original)

# Reference accuracy for Reduced Data
reference_model_reduced = DummyClassifier(strategy='most_frequent')
reference_model_reduced.fit(X_train_reduced_scaled, y_train_reduced)
y_pred_reference_reduced = reference_model_reduced.predict(X_test_reduced_scaled)
accuracy_reference_reduced = accuracy_score(y_test_reduced, y_pred_reference_reduced)

# Define the cost matrix
cost_matrix = {
    1: {1: 1e-6, 2: 1},
    2: {1: 5, 2: 1e-6}
}

# Iterating to see which kernel gives the best result for Original Data
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
score_list_original = {}

for k in kernels:
    # Calculate sample weights based on the cost matrix
    sample_weights = np.array([cost_matrix[yi][yj] for yi, yj in zip(y_train_original, y_train_original)])

    svm_model_original = SVC(random_state=42, kernel=k)
    svm_model_original.fit(X_train_original_scaled, y_train_original)
    #svm_model_original.fit(X_train_original_scaled, y_train_original, sample_weight=sample_weights)
    f_score_original = svm_model_original.score(X_test_original_scaled, y_test_original)
    score_list_original.update({k: f_score_original})

max_val_original = max(score_list_original.values())
list_values_original = list(score_list_original.values())
list_keys_original = list(score_list_original.keys())

# Best kernel for Original Data
best_kernel_original = list_keys_original[list_values_original.index(max_val_original)]

# SVM on Original Data with the best kernel
svm_original_best_kernel = SVC(random_state=42, kernel=best_kernel_original, C=1.0)
svm_original_best_kernel.fit(X_train_original_scaled, y_train_original)
#svm_original_best_kernel.fit(X_train_original_scaled, y_train_original, sample_weight=sample_weights)
# Make predictions for Original Data
y_pred_original_best_kernel = svm_original_best_kernel.predict(X_test_original_scaled)

# Evaluate the model on the original data
accuracy_original_best_kernel = accuracy_score(y_test_original, y_pred_original_best_kernel)
conf_matrix_original_best_kernel = confusion_matrix(y_test_original, y_pred_original_best_kernel)

# Display results for Original Data with the best kernel
print(f"SVM aplicat pe datele originale cu cel mai bun kernel ('{best_kernel_original}'):")
print("Precizie:", accuracy_original_best_kernel)
print("Precizia de referință:", accuracy_reference_original)
disp_original_best_kernel = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_original_best_kernel, display_labels=['Bun', 'Rău'])
disp_original_best_kernel.plot()
plt.title('Matricea de confuzie pentru datele originale')
plt.show()

# Permutation feature importance for Original Data
perm_importance_original = permutation_importance(svm_original_best_kernel, X_train_original_scaled, y_train_original, n_repeats=30, random_state=42)
svm_importance_original = pd.Series(perm_importance_original.importances_mean, index=X_train_original.columns)
svm_importance_original /= svm_importance_original.sum()

# Plot feature importance for Original Data
svm_importance_original.plot(kind='barh', title='Importanța caracteristicilor pentru SVM (Date originale)')
plt.xlabel('Importanță')
plt.ylabel('Caracteristici')
plt.show()

# Iterating to see which kernel gives the best result for Reduced Data
score_list_reduced = {}

for k in kernels:
    # Calculate sample weights based on the cost matrix
    sample_weights_reduced = np.array([cost_matrix[yi][yj] for yi, yj in zip(y_train_reduced, y_train_reduced)])
    
    svm_model_reduced = SVC(random_state=42, kernel=k)
    svm_model_reduced.fit(X_train_reduced_scaled, y_train_reduced)
   #svm_model_reduced.fit(X_train_reduced_scaled, y_train_reduced, sample_weight=sample_weights_reduced)
    f_score_reduced = svm_model_reduced.score(X_test_reduced_scaled, y_test_reduced)
    score_list_reduced.update({k: f_score_reduced})

max_val_reduced = max(score_list_reduced.values())
list_values_reduced = list(score_list_reduced.values())
list_keys_reduced = list(score_list_reduced.keys())

# Best kernel for Reduced Data
best_kernel_reduced = list_keys_reduced[list_values_reduced.index(max_val_reduced)]

# SVM on Reduced Data with the best kernel
svm_reduced_best_kernel = SVC(random_state=42, kernel=best_kernel_reduced, C=1.0)
svm_reduced_best_kernel.fit(X_train_reduced_scaled, y_train_reduced)
#svm_reduced_best_kernel.fit(X_train_reduced_scaled, y_train_reduced, sample_weight=sample_weights_reduced)

# Make predictions for Reduced Data
y_pred_reduced_best_kernel = svm_reduced_best_kernel.predict(X_test_reduced_scaled)

# Evaluate the model on the reduced data
accuracy_reduced_best_kernel = accuracy_score(y_test_reduced, y_pred_reduced_best_kernel)
conf_matrix_reduced_best_kernel = confusion_matrix(y_test_reduced, y_pred_reduced_best_kernel)

# Display results for Reduced Data with the best kernel
print(f"SVM aplicat pe datele reduse cu cel mai bun kernel ('{best_kernel_reduced}'):")
print("Precizie:", accuracy_reduced_best_kernel)
print("Precizia de referință:", accuracy_reference_reduced)
disp_reduced_best_kernel = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_reduced_best_kernel, display_labels=['Bun', 'Rău'])
disp_reduced_best_kernel.plot()
plt.title('Matricea de confuzie pentru datele reduse')
plt.show()

# Permutation feature importance for Reduced Data
perm_importance_reduced = permutation_importance(svm_reduced_best_kernel, X_train_reduced_scaled, y_train_reduced, n_repeats=30, random_state=42)
svm_importance_reduced = pd.Series(perm_importance_reduced.importances_mean, index=X_train_reduced.columns)
svm_importance_reduced /= svm_importance_reduced.sum()

# Plot feature importance for Reduced Data
svm_importance_reduced.plot(kind='barh', title='Importanța caracteristicilor pentru SVM (Date reduse)')
plt.xlabel('Importanță')
plt.ylabel('Caracteristici')
plt.show()
