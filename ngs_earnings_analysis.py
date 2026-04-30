"""
Predicting Post-Graduation Earnings: An Analysis of the National Graduates Survey 2020

This script analyzes the relationship between student debt and post-graduation
earnings for Canadian graduates, using the 2020 National Graduates Survey (NGS).

Originally completed as a research project for ECO225 (Big-Data Tools for
Economists, University of Toronto, Fall 2025). This is a v2 rebuild with
improved methodology:
  - Income and debt are treated as ordinal categorical variables (which they
    are in the survey), not converted to fake-continuous midpoints.
  - The main regression uses ordered logistic regression, which is the
    appropriate model for an ordered outcome like income brackets.
  - Controls include gender, degree level, region, field of study, age at
    graduation, full-time work status, and visible minority status.
  - The classification model uses cross-validation to select the best k for
    KNN, and compares it against logistic regression as a baseline.

Author: Nevashan Baskaran
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------------------------------------------------------
# 1. Load and prepare the data
# -----------------------------------------------------------------------------

# The NGS 2020 file uses Statistics Canada variable codes (CERTLEVP, OWESLGD, etc.)
# We rename the variables we care about to readable names before doing anything else.

DATA_PATH = "NGS2020.csv"   # update this path if running locally
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} graduate records with {df.shape[1]} variables.\n")

# Rename the variables we will use. Codes documented in the NGS 2020 codebook.
rename_map = {
    'CERTLEVP':  'degree_level',        # 1=College, 2=Bachelor's, 3=Master's/Doctorate, 9=Not stated
    'STULOANS':  'has_student_loan',    # 1=Yes, 2=No, 9=Not stated
    'OWESLGD':   'debt_at_grad',        # 0=$0 ... 4=$25K+, 6=Valid skip, 9=Not stated
    'OWEGVIN':   'debt_at_interview',   # same coding as debt_at_grad
    'PERSINCP':  'personal_income',     # 1=<$30K ... 5=$90K+, 99=Not stated
    'GENDER2':   'gender',              # 1=Men+, 2=Women+
    'REG_RESP':  'region',              # 1=Atlantic, 2=Quebec, 3=Ontario, 4=West, 9=Not stated
    'GRADAGEP':  'grad_age',            # 1=<25, 2=25-29, 3=30-39, 4=40+, 9=Not stated
    'PGMCIPAP':  'field_of_study',      # 1-10=field categories, 99=Not stated
    'LFWFTPTP':  'work_ft_pt',          # 1=Full-time, 2=Part-time, 6=Valid skip, 9=Not stated
    'VISBMINP':  'visible_minority',    # 1=Yes, 2=No, 9=Not stated
}
df = df.rename(columns=rename_map)
df = df[list(rename_map.values())]      # keep only the columns we need

# Replace "Not stated" / "Valid skip" codes with NaN so they get dropped cleanly.
# Different variables use different missing codes, which is why we're explicit.
missing_codes = {
    'degree_level':      [9],
    'has_student_loan':  [9],
    'debt_at_grad':      [6, 9],   # 6 = "valid skip" (no loan, so no debt amount)
    'debt_at_interview': [6, 9],
    'personal_income':   [99],
    'region':            [9],
    'grad_age':          [9],
    'field_of_study':    [99],
    'work_ft_pt':        [6, 9],
    'visible_minority':  [9],
}
for col, codes in missing_codes.items():
    df.loc[df[col].isin(codes), col] = np.nan

# For graduates with no student loan, recode debt_at_grad = 0 (rather than NaN).
# This preserves them in the analysis instead of dropping them.
no_loan = df['has_student_loan'] == 2
df.loc[no_loan, 'debt_at_grad'] = 0
df.loc[no_loan, 'debt_at_interview'] = 0


# -----------------------------------------------------------------------------
# 2. Create readable labels for descriptive statistics and plots
# -----------------------------------------------------------------------------

# These dictionaries are only used for display. The regressions below use
# the original numeric codes (or dummies) so that the model interpretation
# stays clean.

degree_labels = {1: "College/Trades", 2: "Bachelor's", 3: "Master's/Doctorate"}
gender_labels = {1: "Men+", 2: "Women+"}
income_labels = {1: "<$30K", 2: "$30-50K", 3: "$50-70K", 4: "$70-90K", 5: "$90K+"}
debt_labels   = {0: "$0", 1: "<$5K", 2: "$5-10K", 3: "$10-25K", 4: "$25K+"}
region_labels = {1: "Atlantic", 2: "Quebec", 3: "Ontario", 4: "West"}

df['degree_label'] = df['degree_level'].map(degree_labels)
df['gender_label'] = df['gender'].map(gender_labels)
df['income_label'] = df['personal_income'].map(income_labels)


# -----------------------------------------------------------------------------
# 3. Descriptive statistics
# -----------------------------------------------------------------------------

# Drop respondents missing the outcome (income) for the analysis sample.
sample = df.dropna(subset=['personal_income', 'degree_level', 'gender']).copy()
print(f"Analysis sample: {len(sample):,} graduates.\n")

print("Income distribution by degree level (% within degree):")
print(pd.crosstab(sample['degree_label'], sample['income_label'],
                  normalize='index').round(3) * 100, "\n")

print("Income distribution by gender (% within gender):")
print(pd.crosstab(sample['gender_label'], sample['income_label'],
                  normalize='index').round(3) * 100, "\n")


# -----------------------------------------------------------------------------
# 4. Visualizations
# -----------------------------------------------------------------------------

sns.set_style("whitegrid")

# Plot 1: stacked income distribution by degree level
plt.figure(figsize=(10, 6))
income_by_degree = pd.crosstab(sample['degree_label'], sample['income_label'],
                               normalize='index') * 100
income_by_degree = income_by_degree[list(income_labels.values())]   # column order
income_by_degree = income_by_degree.loc[["College/Trades", "Bachelor's",
                                         "Master's/Doctorate"]]
income_by_degree.plot(kind='bar', stacked=True, colormap='viridis',
                      figsize=(10, 6), ax=plt.gca())
plt.title("Income Distribution by Degree Level (2020 Canadian Graduates)")
plt.xlabel("Degree Level")
plt.ylabel("Percentage of Graduates")
plt.legend(title="Personal Income", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_income_by_degree.png", dpi=120, bbox_inches='tight')
plt.close()

# Plot 2: gender gap in income
plt.figure(figsize=(10, 6))
income_by_gender = pd.crosstab(sample['gender_label'], sample['income_label'],
                               normalize='index') * 100
income_by_gender = income_by_gender[list(income_labels.values())]
income_by_gender.plot(kind='bar', stacked=True, colormap='viridis',
                      figsize=(10, 6), ax=plt.gca())
plt.title("Income Distribution by Gender (2020 Canadian Graduates)")
plt.xlabel("Gender")
plt.ylabel("Percentage of Graduates")
plt.legend(title="Personal Income", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_income_by_gender.png", dpi=120, bbox_inches='tight')
plt.close()

# Plot 3: relationship between debt at graduation and income (heatmap)
debt_income = sample.dropna(subset=['debt_at_grad'])
heatmap_data = pd.crosstab(debt_income['debt_at_grad'].map(debt_labels),
                           debt_income['personal_income'].map(income_labels),
                           normalize='index') * 100
heatmap_data = heatmap_data.reindex(index=list(debt_labels.values()),
                                    columns=list(income_labels.values()))
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': '% of row'})
plt.title("Income Distribution by Debt at Graduation (% within debt bracket)")
plt.xlabel("Personal Income")
plt.ylabel("Debt at Graduation")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_debt_vs_income_heatmap.png", dpi=120, bbox_inches='tight')
plt.close()

print("Plots saved to outputs/.\n")


# -----------------------------------------------------------------------------
# 5. Ordered logistic regression
# -----------------------------------------------------------------------------

# Why ordered logit instead of OLS:
#   The outcome (personal_income) is an ordered categorical variable with 5
#   brackets. Treating those brackets as continuous dollars (using midpoints)
#   throws away information about the bracket structure and can bias estimates.
#   Ordered logit respects the ordering without assuming the gaps between
#   brackets are equal.

# Build the regression sample. Drop rows missing any predictor.
# We also drop the helper label columns so they don't get pulled into
# the dummy-variable selection below.
reg_sample = sample.drop(columns=['degree_label', 'gender_label', 'income_label']).dropna(
    subset=['debt_at_grad', 'gender', 'degree_level', 'region', 'grad_age',
            'field_of_study', 'work_ft_pt', 'visible_minority']
).copy()

# Create dummy variables for categorical predictors. drop_first=True avoids
# perfect collinearity (the dropped category becomes the reference group).
reg_data = pd.get_dummies(
    reg_sample,
    columns=['gender', 'degree_level', 'region', 'grad_age', 'field_of_study',
             'work_ft_pt', 'visible_minority'],
    drop_first=True
)

# Outcome: ordered income brackets (1 to 5)
y = reg_sample['personal_income'].astype(int)

# Predictors: debt at graduation (ordinal, kept as-is) plus all the dummies
predictor_cols = ['debt_at_grad'] + [c for c in reg_data.columns
                                     if c.startswith(('gender_', 'degree_level_',
                                                      'region_', 'grad_age_',
                                                      'field_of_study_',
                                                      'work_ft_pt_',
                                                      'visible_minority_'))]
X = reg_data[predictor_cols].astype(float)

print(f"Regression sample: {len(X):,} graduates, {X.shape[1]} predictors.\n")

# Fit the ordered logit model
ordered_logit = OrderedModel(y, X, distr='logit').fit(method='bfgs', disp=False)
print("=" * 70)
print("ORDERED LOGISTIC REGRESSION: Predicting Personal Income Bracket")
print("=" * 70)
print(ordered_logit.summary())
print()

# How to read this output:
#   Positive coefficients => higher probability of being in a higher income bracket.
#   The "1.0/2.0", "2.0/3.0" etc. rows at the bottom are the cutpoints between
#   adjacent income brackets and aren't substantively interesting.


# -----------------------------------------------------------------------------
# 6. OLS as a sensitivity check
# -----------------------------------------------------------------------------

# We also run an OLS regression treating the income bracket as a numeric
# variable from 1 to 5. This is methodologically less clean (the gaps between
# brackets aren't equal in dollar terms), but it gives us an interpretable R^2
# and lets us report a "share of variance explained" number that's comparable
# to what most undergrad econometrics courses would generate.

X_ols = sm.add_constant(X)
ols_model = sm.OLS(y, X_ols).fit()
print("=" * 70)
print(f"OLS SENSITIVITY CHECK: R-squared = {ols_model.rsquared:.3f}")
print("=" * 70)
print(f"  (Treating income bracket as a 1-5 numeric scale.)")
print(f"  Adjusted R-squared: {ols_model.rsquared_adj:.3f}")
print(f"  F-statistic p-value: {ols_model.f_pvalue:.3e}")
print()
print("Top 10 strongest predictors by absolute t-statistic:")
coef_table = pd.DataFrame({
    'coefficient': ols_model.params,
    't_stat': ols_model.tvalues,
    'p_value': ols_model.pvalues,
}).drop('const')
coef_table['abs_t'] = coef_table['t_stat'].abs()
print(coef_table.sort_values('abs_t', ascending=False).head(10).round(3), "\n")


# -----------------------------------------------------------------------------
# 7. Classification: predicting "high earner" status
# -----------------------------------------------------------------------------

# Define a high earner as someone in the top two income brackets ($70K+).
# This is more defensible than picking a single dollar cutoff because it
# directly maps onto the survey's income brackets (4 = $70-90K, 5 = $90K+).

reg_sample['high_earner'] = (reg_sample['personal_income'] >= 4).astype(int)
y_class = reg_sample['high_earner']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.30, random_state=42, stratify=y_class
)

# 7a. Cross-validated KNN: pick the best k from a grid
print("=" * 70)
print("KNN CLASSIFIER: Cross-Validated Selection of k")
print("=" * 70)
k_values = [3, 5, 7, 9, 13, 19, 25, 35]
cv_scores = {}
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores[k] = scores.mean()
    print(f"  k={k:>2}: 5-fold CV accuracy = {scores.mean():.4f} "
          f"(+/- {scores.std():.4f})")

best_k = max(cv_scores, key=cv_scores.get)
print(f"\nBest k by CV accuracy: {best_k}\n")

knn_final = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
knn_pred = knn_final.predict(X_test)

print(f"KNN (k={best_k}) test set performance:")
print("Confusion matrix:")
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred,
      target_names=['Not high earner', 'High earner']))

# 7b. Logistic regression as a baseline. In economics applications a logistic
# regression is usually preferred over KNN because the coefficients are
# interpretable. Including it here lets us compare KNN's accuracy against a
# more standard model.
print("=" * 70)
print("LOGISTIC REGRESSION: Baseline Classifier")
print("=" * 70)
logit_clf = LogisticRegression(max_iter=2000).fit(X_train, y_train)
logit_pred = logit_clf.predict(X_test)
print("Confusion matrix:")
print(confusion_matrix(y_test, logit_pred))
print(classification_report(y_test, logit_pred,
      target_names=['Not high earner', 'High earner']))


# -----------------------------------------------------------------------------
# 8. Summary
# -----------------------------------------------------------------------------

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"  Sample size: {len(reg_sample):,} graduates")
print(f"  OLS R-squared: {ols_model.rsquared:.3f} "
      f"(income bracket explained as 1-5 scale)")
print(f"  KNN best k: {best_k} with CV accuracy {cv_scores[best_k]:.4f}")
print(f"  Plots saved to: {OUTPUT_DIR}/")
