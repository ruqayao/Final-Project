# Final-Project
# Fitness Tracker Data Analysis
**INFO-B 211: Information Infrastructure II**
Luddy School of Informatics, Computing, and Engineering - Indiana University Indianapolis

---

## Team Members

| Name | Research Question |
|------|-------------------|
| Sebastian Lucio | RQ1 - Which factors are the strongest predictors of calories burned during a workout? |
| Mel-Neiqua Holloway | RQ2 - Do workout habits or biological characteristics have a greater impact on calorie expenditure? |
| Qaliya Omar | RQ3 - How strongly is workout duration related to calories burned? |
| Henry Adeogun | RQ4 - Which workout types are associated with the highest calorie expenditure? |
| Mel-Neiqua Holloway | RQ5 - Does heart rate influence calorie burn more than body characteristics such as BMI? |

---

## Project Overview

This project analyzes fitness tracker data to explore the relationships between biometric measurements, exercise behavior, and calorie expenditure. Using Python-based data analysis and machine learning, the group investigates five research questions that together address the central theme: **to what extent do biological characteristics influence calorie expenditure compared to workout habits?**

---

## Dataset

### Dataset - Calorie Burnt 15k (HuggingFace)
The dataset was sourced from HuggingFace and consists of two files that are merged on `User_ID`:

| File | Contents |
|------|----------|
| `raw_exercise.csv` | User demographics and biometric measurements (Age, Height, Weight, Duration, Heart Rate, Body Temp, Gender) |
| `raw_calories.csv` | Calorie burn totals per user session |

- **Records:** 15,000 exercise sessions
- **Missing values:** 0
- **Merge key:** `User_ID`

### Original Dataset - Gym Members Exercise Tracking (Kaggle)
The group's original dataset (`gym_members_exercise_tracking_synthetic_data.csv`) was explored during early analysis but replaced due to a data quality issue: the `Calories_Burned` column was synthetically generated independently of all other variables, resulting in near-zero correlations (max r = 0.06) and negative R² scores across all models. The original dataset may still be referenced by other team members' sections.

---

## Research Questions & Methods

**RQ1 - Strongest Predictors of Calories Burned**
Train Linear Regression and Decision Tree Regressor models on biometric and exercise data. Compare feature importance scores and coefficients. Evaluate with R² and MSE.

**RQ2 - Workout Habits vs. Biological Characteristics**
Build two separate models - one using only biological variables (Age, BMI, Height, Weight) and one using only workout variables (Duration, BPM, Workout Type) - and compare their R² scores.

**RQ3 - Workout Duration vs. Calories**
Calculate the Pearson correlation coefficient between session duration and calories burned. Visualize with a scatter plot and regression line. Run a simple linear regression.

**RQ4 - Calorie Burn by Workout Type**
Group the dataset by workout type and calculate average calories burned per category. Visualize using a bar chart and boxplot. Test statistical significance with a one-way ANOVA.

**RQ5 - Heart Rate vs. BMI as Predictors**
Compare the Pearson correlations of Avg BPM and BMI with calories burned. Include both in a multiple regression model to evaluate their relative impact.

---

## Libraries Used

This project uses the following Python libraries covered in the course:

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, merging, and manipulation |
| `numpy` | Numerical operations and array handling |
| `matplotlib` | Base-level data visualization |
| `seaborn` | Statistical visualization (heatmaps, boxplots) |
| `scikit-learn` | Machine learning models, train-test split, metrics |
| `scipy` | Pearson correlation, one-way ANOVA statistical tests |

---

## How to Run

1. Clone the repository
2. Ensure the CSV files are in the same directory as the notebooks, or update the file path in the `pd.read_csv()` call at the top of each notebook
3. Install required libraries if needed:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```
4. Open the desired notebook and run all cells top to bottom

---

## Key Findings (RQ1)

- **Linear Regression R²: 0.9673** and **Random Forest R²: 0.9982** - both models achieve very strong predictive accuracy, confirming that calorie burn is highly predictable from biometric and exercise data
- **Better model: Random Forest** (RMSE of 2.68 calories vs. 11.49 for Linear Regression)
- **Top 3 predictors by feature importance (Random Forest):**
  1. Duration (0.9139) - how long you exercise is by far the biggest driver of calorie burn
  2. Heart Rate (0.0483) - intensity of effort matters more than body composition
  3. Age (0.0263) - demographic factors have a smaller but measurable role
- Weight, Height, and Gender combined account for less than 5% of predictive importance

## Key Findings (RQ3)

- **Correlation Coefficient (r): 0.0286** - Indicates a very weak positive linear relationship between workout duration and calories burned in the main dataset.
- **Linear Regression R²: 0.0008**
- **Reference Data Comparison (R²: 0.9128)** - In contrast, the reference dataset shows a 91% dependency on duration, highlighting a significant discrepancy where the main dataset relies more on other factors like intensity.
- **Predictive Equation:** - Calories = 25.15 X Duration (hrs) + 998.66
- **High Intercept (998.66):** Indicates a high baseline of calories burned regardless of the session length.
- **Low Slope (25.15):** Shows that adding extra time results in a relatively small increase in total burn compared to the baseline.
- **Conclusion:** For this specific gym member population, "time spent" is not the primary driver of expenditure, implying that workout intensity (Heart Rate) or type likely carries more weight.

## Key Findings (RQ4)

- **Dataset:** Gym Members Exercise Tracking (Kaggle), 1,717 records across 4 workout types
- **Average calories burned by type:** Yoga (1,067.47) > Cardio (1,047.76) > Strength (1,016.50) > HIIT (1,005.00)
- **Median calories burned by type:** Yoga (1,070) > Cardio (1,057) > Strength (993) > HIIT (1,005)
- **One-way ANOVA:** F-statistic = 3.19, p-value = 0.023 - differences are statistically significant (p < 0.05)
- **Conclusion:** While statistically significant, the practical difference between workout types is small (~62 calories between Yoga and HIIT). Workout type alone is not a strong driver of calorie burn compared to duration and intensity.

---

## References

- mnemoraorg. (2025). *Calorie Burnt 15k* HuggingFace. https://huggingface.co/datasets/mnemoraorg/calorie-burnt-15k
- Majeed, N. (2025). *Gym Members Exercise Tracking Synthetic Dataset* Kaggle. https://www.kaggle.com/datasets/nadeemajeedch/fitness-tracker-dataset
