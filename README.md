# HR Employee Attrition Analytics

An end-to-end Data Analytics project predicting employee turnover using machine learning. Built as part of the **NextGen Analytics Challenges 2025** competition.

## Table of Contents

- [Overview](#overview)
- [Pipeline](#end-to-end-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing &amp; Feature Engineering](#preprocessing--feature-engineering)
- [Experiment Results](#experiment-results)
- [Conclusion](#conclusion)
- [Project Structure](#project-structure)

---

## Overview

**Objective:** Predict whether an employee will leave the company (attrition) based on various HR features, enabling proactive retention strategies.

**Dataset:**

- **Source:** NextGen Analytics Challenges 2025 - HR Theme
- **Size:** 59,598 records, 23 features
- **Features:** 7 numerical + 16 categorical
- **Target:** `Attrition` (Stayed: 52.5% / Left: 47.5%) - well-balanced

**Tech Stack:**

| Category            | Tools                                     |
| ------------------- | ----------------------------------------- |
| Language            | Python 3.x                                |
| Data Processing     | Pandas, NumPy                             |
| Visualization       | Matplotlib, Seaborn, Power BI             |
| Machine Learning    | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Statistical Testing | SciPy (t-test, Chi-square)                |
| Environment         | Jupyter Notebook                          |

---

## End-to-End Pipeline

```
Data Collection --> EDA --> Preprocessing --> Feature Engineering --> Feature Selection --> Modeling --> Evaluation
     |               |          |                   |                     |                  |            |
   Raw CSV      Univariate   Cleaning          New features        3 Scenarios         11 Models    F2-Score
   59,598       Bivariate    Outliers           Interactions        (Full/FE/Stat)      2 Stages     (Primary)
   records      Multivariate Missing values     Encodings                               Screening
                Hypothesis                                                              + 5-Fold CV
```

| Step             | Description                                                         | Notebook                             |
| ---------------- | ------------------------------------------------------------------- | ------------------------------------ |
| 1. EDA           | Explore distributions, relationships, and patterns                  | `notebooks/01_EDA.ipynb`           |
| 2. Preprocessing | Clean data, handle outliers, encode features, engineer new features | `notebooks/02_Preprocessing.ipynb` |
| 3. Modeling      | Train 11 models across 3 scenarios, evaluate with F2-Score          | `notebooks/03_Modeling.ipynb`      |

---

## Exploratory Data Analysis

> Full analysis in [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb)

### Target Distribution

The target variable `Attrition` is relatively balanced:

- **Stayed:** 52.5% (31,280 employees)
- **Left:** 47.5% (28,318 employees)

This balance means no special sampling techniques (SMOTE, undersampling) are required.

<!-- <chèn ảnh vào đây: reports/figures/target_distribution.png (Pie chart hoặc Count plot của Attrition)> -->

<p align="center">
  <img src="reports/figures/target_distribution.png" width="75%" />
</p>

### Univariate Analysis

**Numerical Features:**

- `Age`: Uniformly distributed (18-59), mean ~38.6
- `Monthly Income`: Approximately normal, range 1,316 - 13,713, mean ~7,302
- `Years at Company`: Right-skewed, 1-47 years, mean ~15.7
- `Distance from Home`: Uniformly distributed, 1-99 km

<p align="center">
  <img src="reports/figures/univariate_numerical.png" width="90%" height="90%" />
</p>

**Categorical Features:**

- `Gender`: Nearly balanced (Male/Female)
- `Job Role`: 5 categories (Education, Media, Healthcare, Technology, Finance)
- `Overtime`: ~40% work overtime
- `Job Level`: Entry / Mid / Senior

<!-- <chèn ảnh vào đây: reports/figures/univariate_numerical.png (Histogram của Age/Monthly Income) và reports/figures/univariate_categorical.png (Bar chart của Job Role/Overtime)> -->

<p align="center">
  <img src="reports/figures/ordinal_univariate_categorical.png" width="95%" />
</p>

### Bivariate & Multivariate Analysis

#### Numerical Variables vs Attrition

<p align="center">
  <img src="reports/figures/numerical_attrition.png" width="90%" />
</p>

- Young employees (20-30 years old) show a significantly higher tendency to leave, whereas stability increases after age 35.
- Regarding commute: retained employees primarily live close to the company (under 40km), while those who left are heavily concentrated at distances over 50km, confirming that geographic location is a major barrier to long-term commitment.
- For work-related factors, **Years at Company** indicates that the first 0-5 years are the most "sensitive" period with a high turnover rate, requiring the organization to focus heavily on the onboarding process and early talent retention.
- **Monthly Income** shows no clear difference between the two groups (the trend lines almost overlap), suggesting that the decision to leave in this dataset is not primarily driven by absolute salary figures.

#### Categorical Variables vs Attrition

**Group 1: Work Environment & Pressure**

<p align="center">
  <img src="reports/figures/work_environment_pressure.png" width="90%" />
</p>

- **Overtime:** While the non-overtime group shows a higher tendency to stay, the balance reverses for the overtime group, where the number of employees leaving (10,025) exceeds those staying (9,425). This confirms that workload pressure is a direct catalyst pushing employees to leave the organization.
- **Work-Life Balance:** The "Good" and "Excellent" groups exhibit strong retention rates. However, alarmingly, in the two lowest categories ("Poor" and "Fair"), the number of departing employees significantly outweighs those staying. Particularly, the "Fair" group sees a massive volume of departures (10,370).
- **Remote Work:** This is the most polarizing factor in this group. Inflexible work policies (No Remote) trigger a massive wave of resignations. Conversely, the remote-enabled group (Yes) demonstrates exceptionally high stability, with retained employees outnumbering departures by nearly 3 to 1, proving that remote work is a "golden key" for retention.
- **Job Satisfaction:** Although the general trend suggests higher satisfaction leads to lower attrition, an interesting paradox exists at the "Very High" level. Here, the number of departing employees (6,405) slightly exceeds those staying (5,706). This implies that this cohort may not leave due to dissatisfaction but rather because they are top performers finding better career advancement opportunities elsewhere.

**Group 2: Motivation & Growth**

<p align="center">
  <img src="reports/figures/motivation_growth.png" width="90%" />
</p>

- **Leadership & Innovation Opportunities:** A concerning reality is that the vast majority of employees lack access to leadership or innovation roles. Although in the "No" group the number of retained employees slightly edges out departures, the absolute volume of turnover here is massive (over 27,000 employees), indicating that a lack of development space is a significant driver of attrition.
- **Job Level:** The **Entry-level** cohort is in a "red alert" state, with the number of departures (15,082) nearly doubling those retained (8,785). In contrast, retention stabilizes dramatically at the **Senior** level. This suggests the company is struggling to retain young/new talent during their initial stages.
- **Performance Rating:** A higher departure rate than retention rate only appears in the low-performance groups (**"Low"** and **"Below Average"**). Meanwhile, **"High"** and **"Average"** performance groups continue to show strong commitment.
- **Employee Recognition:** Recognition levels do not create as drastic a gap in turnover rates as other variables. However, the group receiving **"Low"** recognition still contributes the largest absolute number of departures, indicating that improving the reward culture could help mitigate overall personnel fluctuation.

**Group 3: Organizational Factors**

<p align="center">
  <img src="reports/figures/organization_factor.png" width="90%" />
</p>

- **Company Reputation:** The data reveals a stark polarization: while the **"Good"** and **"Excellent"** reputation groups boast strong retention, the balance immediately flips for the **"Fair"** and **"Poor"** groups. At these two lowest tiers, departing employees outnumber those staying, confirming that declining trust in the organization is a major driver of turnover.
- **Company Size:** **"Small"** companies face the most precarious situation, with departure and retention rates nearly equal. Conversely, **"Medium"** and **"Large"** companies exhibit better retention capabilities, likely due to more robust welfare systems and established operational processes.

**Group 4: Demographics & Profile**

<p align="center">
  <img src="reports/figures/demographic_profile.png" width="90%" />
</p>

- **Marital Status:** This is the sharpest risk-categorization factor. The **Single** group faces an exceptionally high turnover rate (66.8%), completely dominating the **Married** and **Divorced** cohorts.
- **Job Role:** Attrition rates across all departments (Finance, Education, Technology, etc.) hover around 47-48% without major discrepancies. This suggests that the root causes of turnover are **systemic** across the entire company rather than localized to specific departments.
- **Gender:** A notable disparity exists where **Female** employees exhibit a significantly higher turnover rate (53.2%) compared to Males (42.9%). This serves as a signal for the HR department to review diversity, equity, and inclusion (DEI) policies or female-specific benefits.
- **Education Level:** While turnover rates for education levels ranging from High School to Master's are comparably high (~49%), the **PhD** group stands out as a brilliant exception with a retention rate of **75.1%**. It is highly probable that this cohort of senior experts holds stable positions with more specialized compensation packages.

### Power BI Dashboard

🔗 **Live Dashboard:** [View on Power BI Service](https://app.powerbi.com/view?r=eyJrIjoiYWIxNGM0OTYtYjdjZS00OGU1LWJlM2YtNWFiYWEyMTBjOTZkIiwidCI6IjZhYzJhZDA2LTY5MmMtNDY2My1iN2FmLWE5ZmYyYTg2NmQwYyIsImMiOjEwfQ%3D%3D)

![Power BI Dashboard Overview](reports/powerbi/Attrition_Overview.png)

---

## Preprocessing & Feature Engineering

> Full pipeline in [`notebooks/02_Preprocessing.ipynb`](notebooks/02_Preprocessing.ipynb)

### Data Cleaning

| Check            | Result                                                        |
| ---------------- | ------------------------------------------------------------- |
| Missing Values   | None detected (standard + extended NaN check)                 |
| Duplicate Rows   | 0 duplicates                                                  |
| Invalid Values   | All numerical ranges valid, all categorical values consistent |
| Constant Columns | None                                                          |

### Outlier Detection & Treatment

Using the **IQR Method**:

| Feature          | Outliers Detected | Treatment               |
| ---------------- | ----------------- | ----------------------- |
| Years at Company | 273               | Capping (Winsorization) |
| Monthly Income   | 50                | Capping (Winsorization) |
| Other features   | 0                 | No treatment needed     |

<!-- <chèn ảnh vào đây: reports/figures/outliers_boxplot.png (Boxplot của Years at Company và Monthly Income trước/sau khi cap)> -->

Before using Winsorization:

![Outliers Boxplot](reports/figures/before_outliers_boxplot.png)

After using Winsorization:

![Outliers Boxplot](reports/figures/after_outliers_boxplot.png)

### Feature Engineering

New features created through:

- Ordinal encoding for ordered categorical variables
- One-hot encoding for nominal variables
- Feature interactions and domain-specific transformations

### Feature Selection

To reduce dimensionality and remove highly correlated features, we used two main techniques:

- **Correlation Matrix**: Analyzed and removed numerical features with very high mutual correlation to prevent multicollinearity.
- **Feature Importance**: Trained a preliminary Random Forest model to rank features by their importance in predicting Attrition, filtering out the lowest contributing features.

<!-- <chèn ảnh vào đây: reports/figures/feature_importance.png (Biểu đồ Feature Importance từ Random Forest)> -->

<p align="center">
  <img src="reports/figures/feature_importance.png" width="90%" />
</p>

### Hypothesis Testing

Statistical tests to identify significant features:

- **T-test** for numerical features vs Attrition
- **Chi-square test** for categorical features vs Attrition

### Experimental Scenarios

Three feature selection strategies were designed to compare approaches:

| Scenario    | Name            | Features | Method                                     | Purpose                         |
| ----------- | --------------- | -------- | ------------------------------------------ | ------------------------------- |
| **1** | Baseline (Full) | 27       | All original features after encoding       | Baseline comparison             |
| **2** | FE + Selection  | 20       | Correlation + Feature Importance filtering | Evaluate FE effectiveness       |
| **3** | Statistical     | 11       | t-test + Chi-square top features           | Statistical approach comparison |

### Data Splitting

- **Train-Test Split:** 70:30 ratio (stratified)
- **K-Fold CV:** 5-fold stratified cross-validation

Output files organized in `data/scenarios/` with consistent `train_test/` and `kfold/` subdirectories.

---

## Experiment Results

> Full results in [`notebooks/03_Modeling.ipynb`](notebooks/03_Modeling.ipynb)

### Experimental Design

- **Total experiments:** 11 models x 3 scenarios x 2 evaluation stages = 66 experiments
- **Primary Metric:** F2-Score (prioritizes Recall over Precision - minimize False Negatives)
- **Rationale:** Missing an at-risk employee (FN) is more costly than unnecessary retention intervention (FP)

### Models Evaluated

| Group                             | Models                                                            |
| --------------------------------- | ----------------------------------------------------------------- |
| **Linear** (4)              | Logistic Regression, Ridge Classifier, SGD Classifier, Linear SVC |
| **Tree-based** (3)          | Decision Tree, Extra Trees, Gradient Boosting                     |
| **Ensemble & Boosting** (4) | Random Forest, XGBoost, LightGBM, CatBoost                        |

### Two-Stage Evaluation Strategy

| Stage                        | Method           | Purpose                                              |
| ---------------------------- | ---------------- | ---------------------------------------------------- |
| **Stage 1: Screening** | Train-Test Split | Screen all 11 models, select top finalists per group |
| **Stage 2: Final**     | 5-Fold CV        | Robust evaluation of finalists, select champion      |

### Results Summary (5-Fold CV - Stage 2)

**Scenario 1 - Baseline (Full Features, 27 features):**

| Rank | Model              | F2 (mean +/- std)           | Recall | Precision | Accuracy |
| ---- | ------------------ | --------------------------- | ------ | --------- | -------- |
| 1    | **LightGBM** | **0.7437 +/- 0.0042** | 0.7439 | 0.7427    | 0.7632   |
| 2    | CatBoost           | 0.7423 +/- 0.0037           | 0.7421 | 0.7430    | 0.7602   |
| 3    | Gradient Boosting  | 0.7397 +/- 0.0039           | 0.7379 | 0.7468    | 0.7648   |

**Scenario 2 - Feature Engineering + Selection (20 features):**

| Rank | Model              | F2 (mean +/- std)           | Recall | Precision | Accuracy |
| ---- | ------------------ | --------------------------- | ------ | --------- | -------- |
| 1    | **LightGBM** | **0.7389 +/- 0.0041** | 0.7398 | 0.7462    | 0.7574   |
| 2    | Gradient Boosting  | 0.7363 +/- 0.0041           | 0.7355 | 0.7486    | 0.7574   |
| 3    | CatBoost           | 0.7359 +/- 0.0031           | 0.7365 | 0.7453    | 0.7557   |

**Scenario 3 - Statistical Features (11 features):**

| Rank | Model                       | F2 (mean +/- std)           | Recall | Precision | Accuracy |
| ---- | --------------------------- | --------------------------- | ------ | --------- | -------- |
| 1    | **Gradient Boosting** | **0.7338 +/- 0.0041** | 0.7352 | 0.7373    | 0.7501   |
| 2    | LightGBM                    | 0.7333 +/- 0.0026           | 0.7348 | 0.7398    | 0.7530   |
| 3    | CatBoost                    | 0.7289 +/- 0.0031           | 0.7295 | 0.7335    | 0.7472   |

### Key Findings

1. **LightGBM** and **CatBoost** consistently perform as top models across all scenarios
2. **Scenario 1** (full features) produces the best overall results, suggesting the dataset benefits from more feature information
3. Performance degradation from Scenario 1 to Scenario 3 is moderate (~1%), indicating core predictive features are well-captured by statistical selection
4. All top models show **low variance** across folds (std < 0.005), indicating stable and reliable predictions
5. **Boosting methods** (LightGBM, CatBoost, Gradient Boosting, XGBoost) consistently outperform bagging and linear approaches

---

## Conclusion

- **Best Model:** LightGBM with full features (Scenario 1) achieves **F2 = 0.7437** with high stability
- **Trade-off:** Reducing features from 27 to 11 only drops F2 by ~1.4%, offering a simpler, more interpretable model
- **Business Impact:** The model can identify ~74% of at-risk employees, enabling targeted retention programs

---

## Project Structure

```
hr_attrition_analytics/
|-- README.md
|-- LICENSE
|-- .gitignore
|-- data/
|   |-- raw/                           # Original dataset
|   |-- processed/                     # Cleaned & feature-engineered data
|   |-- scenarios/                     # 3 experimental scenarios
|       |-- scenario1/                 # Baseline (Full Features)
|       |-- scenario2/                 # FE + Selection
|       |-- scenario3/                 # Statistical Features
|-- notebooks/
|   |-- 01_EDA.ipynb                   # Exploratory Data Analysis
|   |-- 02_Preprocessing.ipynb         # Data Cleaning & Feature Engineering
|   |-- 03_Modeling.ipynb              # Model Training & Evaluation
|-- reports/
|   |-- figures/                       # EDA visualizations
|   |-- powerbi/                       # Power BI dashboard (coming soon)
|-- docs/                              # Problem statement & references
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
