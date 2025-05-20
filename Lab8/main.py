import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.decomposition import PCA

# 1. Завантаження даних
df = pd.read_csv("blood.csv")
X = df.drop(columns="Class")
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Базові моделі
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 3. Підбір гіперпараметрів Logistic Regression
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(class_weight='balanced', random_state=42))
])

lr_params = {
    'lr__C': np.logspace(-2, 2, 10),
    'lr__penalty': ['l2'],
    'lr__fit_intercept': [True, False]
}

lr_grid = GridSearchCV(lr_pipe, lr_params, scoring='f1', cv=5)
lr_grid.fit(X_train, y_train)
best_lr = lr_grid.best_estimator_
y_pred_lr = best_lr.predict(X_test)

print("\n=== Logistic Regression (GridSearchCV) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1-score:", f1_score(y_test, y_pred_lr))
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Logistic Regression (GridSearchCV)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 4. Створення нових ознак (Polynomial Features)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42, stratify=y
)

poly_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', random_state=42))
])
poly_model.fit(X_poly_train, y_poly_train)
y_pred_poly = poly_model.predict(X_poly_test)

print("\n=== Logistic Regression + Polynomial Features ===")
print("Accuracy:", accuracy_score(y_poly_test, y_pred_poly))
print("F1-score:", f1_score(y_poly_test, y_pred_poly))
print("Classification Report:")
print(classification_report(y_poly_test, y_pred_poly))
cm_poly = confusion_matrix(y_poly_test, y_pred_poly)
sns.heatmap(cm_poly, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Logistic Regression + Polynomial Features")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 5. Зниження розмірності (PCA)
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3, random_state=42)),
    ('clf', LogisticRegression(class_weight='balanced', random_state=42))
])
pca_pipeline.fit(X_train, y_train)
y_pred_pca = pca_pipeline.predict(X_test)

print("\n=== Logistic Regression + PCA ===")
print("Accuracy:", accuracy_score(y_test, y_pred_pca))
print("F1-score:", f1_score(y_test, y_pred_pca))
print("Classification Report:")
print(classification_report(y_test, y_pred_pca))
cm_pca = confusion_matrix(y_test, y_pred_pca)
sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Logistic Regression + PCA")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 6. Підбір гіперпараметрів Decision Tree
dt_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', DecisionTreeClassifier(random_state=42))
])

dt_params = {
    'clf__max_depth': [2, 4, 6, 8, 10],
    'clf__min_samples_split': [2, 4, 6],
    'clf__min_samples_leaf': [1, 2, 4]
}

dt_grid = GridSearchCV(dt_pipe, dt_params, scoring='f1', cv=5)
dt_grid.fit(X_train, y_train)
y_pred_dt = dt_grid.best_estimator_.predict(X_test)

print("\n=== Decision Tree (GridSearchCV) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("F1-score:", f1_score(y_test, y_pred_dt))
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Best Decision Tree (GridSearchCV)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. Decision Tree + Polynomial Features
dt_poly_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', DecisionTreeClassifier(random_state=42))
])
dt_poly_model.fit(X_poly_train, y_poly_train)
y_pred_dt_poly = dt_poly_model.predict(X_poly_test)

print("\n=== Decision Tree + Polynomial Features ===")
print("Accuracy:", accuracy_score(y_poly_test, y_pred_dt_poly))
print("F1-score:", f1_score(y_poly_test, y_pred_dt_poly))
print("Classification Report:")
print(classification_report(y_poly_test, y_pred_dt_poly))
cm_dt_poly = confusion_matrix(y_poly_test, y_pred_dt_poly)
sns.heatmap(cm_dt_poly, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Decision Tree + Polynomial Features")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8. Decision Tree + PCA
dt_pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3, random_state=42)),
    ('clf', DecisionTreeClassifier(random_state=42))
])
dt_pca_pipeline.fit(X_train, y_train)
y_pred_dt_pca = dt_pca_pipeline.predict(X_test)

print("\n=== Decision Tree + PCA ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt_pca))
print("F1-score:", f1_score(y_test, y_pred_dt_pca))
print("Classification Report:")
print(classification_report(y_test, y_pred_dt_pca))
cm_dt_pca = confusion_matrix(y_test, y_pred_dt_pca)
sns.heatmap(cm_dt_pca, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Decision Tree + PCA")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9. Підбір гіперпараметрів Random Forest
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

rf_params = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [None, 4, 8],
    'clf__min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(rf_pipe, rf_params, scoring='f1', cv=5)
rf_grid.fit(X_train, y_train)
y_pred_rf = rf_grid.best_estimator_.predict(X_test)

print("\n=== Random Forest (GridSearchCV) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1-score:", f1_score(y_test, y_pred_rf))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Best Random Forest (GridSearchCV)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 10. Random Forest + Polynomial Features
rf_poly_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])
rf_poly_model.fit(X_poly_train, y_poly_train)
y_pred_rf_poly = rf_poly_model.predict(X_poly_test)

print("\n=== Random Forest + Polynomial Features ===")
print("Accuracy:", accuracy_score(y_poly_test, y_pred_rf_poly))
print("F1-score:", f1_score(y_poly_test, y_pred_rf_poly))
print("Classification Report:")
print(classification_report(y_poly_test, y_pred_rf_poly))
cm_rf_poly = confusion_matrix(y_poly_test, y_pred_rf_poly)
sns.heatmap(cm_rf_poly, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Random Forest + Polynomial Features")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 11. Random Forest + PCA
rf_pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3, random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])
rf_pca_pipeline.fit(X_train, y_train)
y_pred_rf_pca = rf_pca_pipeline.predict(X_test)

print("\n=== Random Forest + PCA ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_pca))
print("F1-score:", f1_score(y_test, y_pred_rf_pca))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf_pca))
cm_rf_pca = confusion_matrix(y_test, y_pred_rf_pca)
sns.heatmap(cm_rf_pca, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Random Forest + PCA")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



