# ============================================================
# A/B TEST ANALYSIS - E-COMMERCE PRICING STRATEGY
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, f_oneway

# -------------------- LOAD DATA --------------------

ab = pd.read_csv(r"C:\Users\Kittu\OneDrive\Desktop\DSML MODULE\ab_data.csv")
countries = pd.read_csv(r"C:\Users\Kittu\OneDrive\Desktop\DSML MODULE\countries.csv")

print("Original data:", ab.shape)

# -------------------- DATA CLEANING --------------------

# Remove group/landing-page mismatches
ab = ab[
    ((ab["group"] == "control") & (ab["landing_page"] == "old_page")) |
    ((ab["group"] == "treatment") & (ab["landing_page"] == "new_page"))
]

# Remove duplicate users
ab = ab.drop_duplicates("user_id")

print("Cleaned data:", ab.shape)

# -------------------- MERGE COUNTRY DATA --------------------

countries = countries.drop_duplicates("user_id")

data = ab.merge(countries, on="user_id", how="inner")

print("\nCountry distribution:")
print(data["country"].value_counts())

# ============================================================
# 1. CONVERSION RATE
# ============================================================

summary = data.groupby("group")["converted"].agg(
    ["count", "sum", "mean"]
)

summary["conversion_%"] = summary["mean"] * 100

print("\nConversion Summary:")
print(summary)

control = data[data["group"] == "control"]["converted"]
treatment = data[data["group"] == "treatment"]["converted"]

control_rate = control.mean()
treatment_rate = treatment.mean()

print("\nControl Conversion :", round(control_rate * 100, 3), "%")
print("Treatment Conversion:", round(treatment_rate * 100, 3), "%")
print("Difference          :", round((treatment_rate-control_rate)*100, 3), "percentage points")

# -------------------- GRAPH --------------------

summary["conversion_%"].plot(
    kind="bar",
    figsize=(7,5)
)

plt.title("Conversion Rate by Pricing Strategy")
plt.xlabel("Group")
plt.ylabel("Conversion Rate (%)")
plt.xticks(rotation=0)
plt.show()

# ============================================================
# 2. WELCH T-TEST
# ============================================================

t_stat, t_p = ttest_ind(
    treatment,
    control,
    equal_var=False
)

print("\nWelch T-Test")
print("t-statistic:", round(t_stat, 4))
print("p-value:", round(t_p, 4))

# ============================================================
# 3. CHI-SQUARE TEST
# ============================================================

table = pd.crosstab(
    data["group"],
    data["converted"]
)

chi2, chi_p, dof, expected = chi2_contingency(table)

print("\nChi-Square Test")
print("Chi-square:", round(chi2, 4))
print("p-value:", round(chi_p, 4))

# ============================================================
# 4. COUNTRY ANALYSIS
# ============================================================

country_summary = data.groupby("country")["converted"].agg(
    ["count", "sum", "mean"]
)

country_summary["conversion_%"] = (
    country_summary["mean"] * 100
)

print("\nCountry Analysis:")
print(country_summary)

# Country graph
country_summary["conversion_%"].plot(
    kind="bar",
    figsize=(7,5)
)

plt.title("Conversion Rate by Country")
plt.xlabel("Country")
plt.ylabel("Conversion Rate (%)")
plt.xticks(rotation=0)
plt.show()

# ============================================================
# 5. ANOVA ACROSS COUNTRIES
# ============================================================

country_groups = [
    group["converted"].values
    for _, group in data.groupby("country")
]

f_country, p_country = f_oneway(*country_groups)

print("\nCountry ANOVA")
print("F-statistic:", round(f_country, 4))
print("p-value:", round(p_country, 4))

# ============================================================
# 6. COUNTRY × STRATEGY ANALYSIS
# ============================================================

cs = data.groupby(
    ["country", "group"]
)["converted"].mean().reset_index()

cs["conversion_%"] = cs["converted"] * 100

print("\nCountry × Strategy:")
print(cs)

# Graph
sns.barplot(
    data=data,
    x="country",
    y="converted",
    hue="group"
)

plt.title("Conversion Rate by Country and Strategy")
plt.xlabel("Country")
plt.ylabel("Conversion Rate")
plt.show()

# ============================================================
# 7. ANOVA: COUNTRY × STRATEGY
# ============================================================

data["country_group"] = (
    data["country"] + "_" + data["group"]
)

six_groups = [
    group["converted"].values
    for _, group in data.groupby("country_group")
]

f_six, p_six = f_oneway(*six_groups)

print("\nCountry × Strategy ANOVA")
print("F-statistic:", round(f_six, 4))
print("p-value:", round(p_six, 4))

# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n" + "="*55)
print("FINAL CONCLUSION")
print("="*55)

if t_p < 0.05:
    print("The pricing strategies show a statistically significant difference.")
else:
    print("No statistically significant difference was found between the pricing strategies.")

if chi_p < 0.05:
    print("Chi-square: Strategy and conversion are significantly associated.")
else:
    print("Chi-square: No significant association between strategy and conversion.")

if p_country < 0.05:
    print("Country has a significant effect on conversion.")
else:
    print("No significant difference in conversion across countries.")

if p_six < 0.05:
    print("Country and strategy groups differ significantly.")
else:
    print("No significant difference among country × strategy groups.")
