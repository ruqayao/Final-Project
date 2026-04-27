# # RQ1: Which factors are the strongest predictors of calories burned during a workout?
# **INFO-B 211: Information Infrastructure II**  
# **Fitness Tracker Data Analysis**
#
# ---
# **Dataset:** Calorie Burnt 15k - real exercise session records for ~15,000 users  
# **Source:** HuggingFace (`raw_exercise.csv` + `raw_calories.csv`, merged on `User_ID`)
#
# **Method:**
# - Merge exercise and calorie datasets on `User_ID`
# - Train Linear Regression and Random Forest Regressor models
# - Compare feature importance / coefficients for key variables
# - Evaluate model performance using R² and MSE

# ## Step 1: Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

sns.set_theme(style='whitegrid')
print('Libraries loaded.')

# ## Step 2: Load & Merge Data

exercise = pd.read_csv('raw_exercise.csv')
calories = pd.read_csv('raw_calories.csv')

print('Exercise shape:', exercise.shape)
print('Calories shape:', calories.shape)
print()
print('Exercise preview:')
print(exercise.head())
print('Calories preview:')
print(calories.head())

# Merge both files on User_ID
df = pd.merge(exercise, calories, on='User_ID')

print('Merged shape:', df.shape)
print('Missing values:')
print(df.isnull().sum())
print(df.head())

# Summary statistics
print(df.describe())

# ## Step 3: Prepare Features & Target

# Encode Gender (male=1, female=0)
le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])
print('Gender mapping:', dict(zip(le.classes_, le.transform(le.classes_))))

feature_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Gender_Encoded']

X = df[feature_cols]
y = df['Calories']

print('\nFeatures shape:', X.shape)
print('Any NaN in X:', X.isnull().any().any())
print('Any NaN in y:', y.isnull().any())

# Correlation of each feature with Calories — preview before modeling
corrs = X.corrwith(y).sort_values(ascending=False)
print('Pearson correlations with Calories:')
print(corrs.round(4))

# Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df[feature_cols + ['Calories']].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5, square=True, ax=ax)
ax.set_title('Correlation Heatmap — All Features vs. Calories', fontsize=13)
plt.tight_layout()
plt.savefig('rq1_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print('Training samples:', len(X_train))
print('Testing samples: ', len(X_test))

# ## Step 4: Train Models

# --- Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

r2_lr  = r2_score(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)

print('Linear Regression')
print(f'  R²:  {r2_lr:.4f}')
print(f'  MSE: {mse_lr:.2f}')

# --- Random Forest Regressor ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

r2_rf  = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

print('Random Forest Regressor')
print(f'  R²:  {r2_rf:.4f}')
print(f'  MSE: {mse_rf:.2f}')

# ## Step 5: Model Performance Comparison

# Summary table
performance = pd.DataFrame({
    'Model':  ['Linear Regression', 'Random Forest'],
    'R²':     [round(r2_lr, 4),  round(r2_rf, 4)],
    'MSE':    [round(mse_lr, 2), round(mse_rf, 2)],
    'RMSE':   [round(np.sqrt(mse_lr), 2), round(np.sqrt(mse_rf), 2)]
})
print(performance.to_string(index=False))

# R² comparison bar chart
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(['Linear Regression', 'Random Forest'],
              [r2_lr, r2_rf],
              color=['steelblue', 'mediumseagreen'],
              edgecolor='white', width=0.45)
ax.set_ylim(0, 1)
ax.set_title('Model Performance: R² Score Comparison', fontsize=13)
ax.set_ylabel('R² Score')

for bar, val in zip(bars, [r2_lr, r2_rf]):
    ax.text(bar.get_x() + bar.get_width()/2, val - 0.05,
            f'{val:.4f}', ha='center', fontsize=12, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('rq1_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ## Step 6: Feature Importance — Decision Tree

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

print('Feature Importances (Random Forest):')
print(importances.round(4).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
importances.plot(kind='barh', color='mediumseagreen', ax=ax, edgecolor='white')
ax.set_title('RQ1: Feature Importance for Calorie Prediction\n(Random Forest Regressor)', fontsize=13)
ax.set_xlabel('Importance Score')
ax.invert_yaxis()

for i, val in enumerate(importances):
    ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('rq1_feature_importance_rf.png', dpi=150, bbox_inches='tight')
plt.show()

# ## Step 7: Feature Coefficients — Linear Regression

coef_df = pd.DataFrame({
    'Feature':     feature_cols,
    'Coefficient': lr.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print('Linear Regression Coefficients (sorted by magnitude):')
print(coef_df.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['steelblue' if c >= 0 else 'tomato' for c in coef_df['Coefficient']]

ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('RQ1: Linear Regression Coefficients\n(Blue = Positive Effect, Red = Negative Effect)', fontsize=13)
ax.set_xlabel('Coefficient Value')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('rq1_lr_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()

# ## Step 8: Top Predictor Deep Dive — Duration vs. Calories

# Scatter plot of the strongest predictor
top_feature = importances.index[0]

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df[top_feature], df['Calories'], alpha=0.2, s=10, color='steelblue')

m, b = np.polyfit(df[top_feature], df['Calories'], 1)
x_line = np.linspace(df[top_feature].min(), df[top_feature].max(), 200)
ax.plot(x_line, m * x_line + b, color='crimson', linewidth=2,
        label=f'r = {df[[top_feature, "Calories"]].corr().iloc[0,1]:.4f}')

ax.set_title(f'Top Predictor: {top_feature} vs. Calories Burned', fontsize=13)
ax.set_xlabel(top_feature)
ax.set_ylabel('Calories Burned')
ax.legend()
plt.tight_layout()
plt.savefig('rq1_top_predictor_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# ## Step 9: Summary

top3 = importances.head(3).index.tolist()
winner = 'Random Forest' if r2_rf > r2_lr else 'Linear Regression'

print('=' * 60)
print('RQ1 SUMMARY: Strongest Predictors of Calories Burned')
print('=' * 60)
print(f"""
Dataset: 15,000 real exercise session records

Model Performance:
  Linear Regression  → R²: {r2_lr:.4f}, MSE: {mse_lr:.2f}
  Random Forest      → R²: {r2_rf:.4f}, MSE: {mse_rf:.2f}
  Better model: {winner}

Top 3 Predictors (Random Forest Feature Importance):
  1. {top3[0]}
  2. {top3[1]}
  3. {top3[2]}

Conclusion:
  Both models achieve high predictive accuracy (R² > 0.96),
  confirming that calorie burn can be reliably predicted from
  biometric and exercise data. {top3[0]} and {top3[1]}
  are by far the strongest drivers, suggesting that workout
  behavior and physiological intensity matter more than
  demographic factors like age or weight alone.
""")

def performance_label(r2):
    if r2 >= 0.9:
        return "Very Strong"
    elif r2 >= 0.8:
        return "Strong"
    elif r2 >= 0.6:
        return "Moderate"
    else:
        return "Weak"

print("\nPerformance Evaluation:")
print(f"Linear Regression: {performance_label(r2_lr)} predictive performance")
print(f"Random Forest: {performance_label(r2_rf)} predictive performance")
