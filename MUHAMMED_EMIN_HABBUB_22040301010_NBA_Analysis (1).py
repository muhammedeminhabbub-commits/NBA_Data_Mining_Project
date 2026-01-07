import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

# 1. Veri Seti Simülasyonu (Kaggle'daki NBA Seasons Stats yapısına uygun)
def generate_nba_data(n_samples=5000):
    np.random.seed(42)
    data = {
        'Year': np.random.randint(1950, 2018, n_samples),
        'Pos': np.random.choice(['PG', 'SG', 'SF', 'PF', 'C'], n_samples),
        'Age': np.random.randint(18, 40, n_samples),
        'G': np.random.randint(1, 83, n_samples),
        'MP': np.random.randint(10, 3000, n_samples),
        'PTS': np.random.randint(0, 2500, n_samples),
        'AST': np.random.randint(0, 1000, n_samples),
        'TRB': np.random.randint(0, 1200, n_samples),
        'STL': np.random.randint(0, 200, n_samples),
        'BLK': np.random.randint(0, 200, n_samples),
        'TOV': np.random.randint(0, 400, n_samples),
        'PER': np.random.normal(15, 5, n_samples)
    }
    df = pd.DataFrame(data)
    # All-Star etiketi oluşturma (PER > 20 ve PTS > 1000 ise 1, değilse 0)
    df['All_Star'] = ((df['PER'] > 20) & (df['PTS'] > 1000)).astype(int)
    return df

df = generate_nba_data(5000) # Hızlı analiz için örneklem sayısını azalttık

# 2. Veri Ön İşleme ve Özellik Mühendisliği
# Filtreleme
df = df[df['G'] >= 10] # En az 10 maç oynayanlar
df = df[df['MP'] >= 500] # En az 500 dakika oynayanlar

# Yeni Özellikler Oluşturma (Feature Engineering)
df['PTS_per_G'] = df['PTS'] / df['G']
df['AST_per_G'] = df['AST'] / df['G']
df['TRB_per_G'] = df['TRB'] / df['G']

# Kategorik verileri dönüştürme (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['Pos'], drop_first=True) # drop_first=True to avoid multicollinearity

# Özellik Tipleri Raporu
feature_types = []
for col in df_encoded.columns:
    if df_encoded[col].dtype in ['int64', 'float64']:
        feature_types.append({'Feature': col, 'Type': 'Numeric'})
    elif df_encoded[col].dtype == 'uint8': # For one-hot encoded columns
        feature_types.append({'Feature': col, 'Type': 'Categorical (One-Hot Encoded)'})
    else:
        feature_types.append({'Feature': col, 'Type': str(df_encoded[col].dtype)})
feature_types_df = pd.DataFrame(feature_types)
feature_types_df.to_csv('feature_types_summary.csv', index=False)

# 3. Sınıflandırma Modelleri (All-Star Tahmini)
X_cls = df_encoded.drop(['All_Star', 'PER', 'Year'], axis=1)
y_cls = df_encoded['All_Star']

# Sınıf Dağılımı (Pie Chart)
plt.figure(figsize=(8, 8))
df['All_Star'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Normal', 'All-Star'], colors=['lightcoral', 'lightskyblue'])
plt.title('All-Star Dağılımı')
plt.ylabel('') # Remove default 'All_Star' label on y-axis
plt.savefig('all_star_distribution.png')
plt.close()

# StratifiedKFold kullanımı
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train-Test Split
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=42)

# Modeller ve Hyperparameter Tuning
# a. Random Forest
rf_params = {'n_estimators': [50], 'max_depth': [10]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=skf, scoring='f1')
rf_grid.fit(X_train_cls, y_train_cls)
best_rf = rf_grid.best_estimator_

# b. Gradient Boosting
gb_params = {'n_estimators': [50], 'learning_rate': [0.1]}
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=skf, scoring='f1')
gb_grid.fit(X_train_cls, y_train_cls)
best_gb = gb_grid.best_estimator_

# c. Stacking
estimators = [('rf', best_rf), ('gb', best_gb)]
stack_cls = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack_cls.fit(X_train_cls, y_train_cls)

# 4. Regresyon Modelleri (PER Tahmini)
X_reg = df_encoded.drop(['PER', 'All_Star', 'Year'], axis=1)
y_reg = df_encoded['PER']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# a. Linear Regression (Base Model)
linear_reg = LinearRegression()
linear_reg.fit(X_train_reg, y_train_reg)

# b. Random Forest Regressor
rf_reg_params = {'n_estimators': [50]}
rf_reg_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_reg_params, cv=5)
rf_reg_grid.fit(X_train_reg, y_train_reg)
best_rf_reg = rf_reg_grid.best_estimator_

# c. Gradient Boosting Regressor
gb_reg_params = {'n_estimators': [50], 'learning_rate': [0.1]}
gb_reg_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_reg_params, cv=5)
gb_reg_grid.fit(X_train_reg, y_train_reg)
best_gb_reg = gb_reg_grid.best_estimator_

# 5. Performans Değerlendirme ve Görselleştirme
# Sınıflandırma Sonuçları
models_cls = {'Random Forest': best_rf, 'Gradient Boosting': best_gb, 'Stacking': stack_cls}
cls_results = []
cls_class_wise_results = []

plt.figure(figsize=(10, 8))
for name, model in models_cls.items():
    y_pred = model.predict(X_test_cls)
    y_prob = model.predict_proba(X_test_cls)[:, 1]
    
    acc = accuracy_score(y_test_cls, y_pred)
    prec = precision_score(y_test_cls, y_pred, zero_division=0)
    rec = recall_score(y_test_cls, y_pred, zero_division=0)
    f1 = f1_score(y_test_cls, y_pred, zero_division=0)
    auc = roc_auc_score(y_test_cls, y_prob)
    
    cls_results.append([name, acc, prec, rec, f1, auc])
    
    # Class-wise metrics
    report = classification_report(y_test_cls, y_pred, output_dict=True, zero_division=0)
    cls_class_wise_results.append({
        'Model': name,
        'Class 0 Precision': report['0']['precision'],
        'Class 0 Recall': report['0']['recall'],
        'Class 0 F1-Score': report['0']['f1-score'],
        'Class 1 Precision': report['1']['precision'],
        'Class 1 Recall': report['1']['recall'],
        'Class 1 F1-Score': report['1']['f1-score']
    })
    
    fpr, tpr, _ = roc_curve(y_test_cls, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('roc_curves.png')
plt.close()

# Regresyon Sonuçları
models_reg = {'Linear Regression': linear_reg, 'Random Forest Reg': best_rf_reg, 'Gradient Boosting Reg': best_gb_reg}
reg_results = []

for name, model in models_reg.items():
    y_pred = model.predict(X_test_reg)
    
    r2 = r2_score(y_test_reg, y_pred)
    mse = mean_squared_error(y_test_reg, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reg, y_pred)
    
    # MAPE hesaplaması: Sıfıra bölme hatasını önlemek için y_test_reg > 0 kontrolü
    mape = np.mean(np.abs((y_test_reg - y_pred) / y_test_reg[y_test_reg != 0])) * 100 if (y_test_reg != 0).any() else 0
    
    reg_results.append([name, r2, mse, rmse, mae, mape])

# Sonuçları Kaydetme
cls_df = pd.DataFrame(cls_results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
reg_df = pd.DataFrame(reg_results, columns=['Model', 'R2', 'MSE', 'RMSE', 'MAE', 'MAPE'])
cls_class_wise_df = pd.DataFrame(cls_class_wise_results)

cls_df.to_csv('classification_results.csv', index=False)
reg_df.to_csv('regression_results.csv', index=False)
cls_class_wise_df.to_csv('classification_class_wise_results.csv', index=False)

print("Analiz tamamlandı. Dosyalar oluşturuldu.")
