# 💳 Financial Transaction Fraud Detection

> An end-to-end fraud analytics and machine learning project focused on
> discovering suspicious transaction patterns, engineering behavioral
> features, and evaluating tree-based classifiers on a highly imbalanced
> financial dataset.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC4E20)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview

This project analyzes financial transaction behavior to identify
patterns associated with fraudulent activity and builds machine learning
models for fraud detection.

The workflow combines:

-   **Feature Engineering** from transaction timestamps and customer
    behavior
-   **Exploratory Data Analysis (EDA)** to uncover temporal, behavioral,
    and amount-related fraud patterns
-   **Decision Tree analysis** to extract interpretable fraud ranges and
    rules
-   **Machine Learning** using Random Forest, AdaBoost, and XGBoost
-   **Threshold analysis** for imbalanced classification
-   **Feature Importance analysis** to identify the strongest predictive
    signals

The modeling dataset is highly imbalanced: only **0.5164%** of the
modeling records are fraudulent. Because of this, metrics such as
**Precision, Recall, F1-score, and ROC-AUC** are more informative than
Accuracy alone.

------------------------------------------------------------------------

## 📊 Dataset Snapshot

After removing rows with missing values in the final modeling features:

  Item                          Value
  --------------------- -------------
  Modeling records          1,851,395
  Normal transactions       1,841,835
  Fraud transactions            9,560
  Fraud rate              **0.5164%**
  Train set                 1,481,116
  Test set                    370,279
  Train/Test split          80% / 20%
  Split strategy           Stratified

The broader transaction dataset used in parts of the Decision Tree
analysis contains **1,852,394 transactions**, including **9,651 fraud
cases**.

------------------------------------------------------------------------

## 🛠️ Feature Engineering

Several analytical features were derived from the raw transaction data.

### ⏰ Time Features

-   `Hour of Day` --- transaction hour from 0 to 23
-   `Day of Week` --- transaction weekday
-   `Month` --- transaction month
-   `Time Interval (Minutes)` --- time since the customer's previous
    transaction
-   `Weekend vs Weekday` --- weekend/weekday indicator

### 👤 Customer Behavior Features

-   `Total Transactions per Customer`
-   `Active Days`
-   `Transactions per Day`

### 💰 Amount Features

-   `Daily Transaction Amount`
-   `Maximum Daily Transaction Amount`

### 🤖 Final ML Features

The final comparison of Random Forest, AdaBoost, and XGBoost used:

``` text
amt
hour
city_pop
lat
long
merch_lat
merch_long
time_diff_minutes
```

Target:

``` text
is_fraud
```

------------------------------------------------------------------------

# 🔎 Exploratory Data Analysis

## Distribution & Skewness

Several transaction and customer-level variables are strongly
**right-skewed**.

-   Customer transaction counts are concentrated around the more common
    activity ranges, with fewer customers at the extremes.
-   Customer active-day counts also show an uneven distribution.
-   Daily transaction amounts and maximum daily transaction amounts are
    heavily concentrated at low values, with relatively few high-value
    observations.
-   Transaction intervals are right-skewed: short intervals occur much
    more frequently than long intervals.

These distributions indicate that robust behavioral features and
non-linear models can be useful for fraud detection.

------------------------------------------------------------------------

## ⏱️ Time-Based Patterns

Daily transaction volume fluctuates substantially from **January 2019 to
January 2021**, while the 7-day moving average remains comparatively
stable.

Notable temporal observations include:

-   **December** has the highest transaction volume, at approximately
    **280,000 transactions**.
-   **February** has the lowest, at approximately **100,000
    transactions**.
-   Fraud rates vary across months rather than following a uniform
    seasonal pattern.
-   Transaction volume is highest on **Monday, Saturday, and Sunday**
    and lowest on **Wednesday**.
-   Weekend and weekday fraud rates are very similar: approximately
    **0.51% vs 0.53%**.
-   Fraud activity is strongly concentrated during specific late-night
    hours.
-   Fraud rate is around **1.4% between 00:00 and 03:00**, falls close
    to zero during much of the day, and reaches approximately **2.6%
    around 21:00--22:00**.

> **Key temporal signal:** the period from approximately **21:00 to
> 03:00** is substantially more fraud-prone than most daytime hours.

------------------------------------------------------------------------

## 👥 Customer Daily Behavior

Most customers perform relatively few transactions per day.

-   The highest-frequency group contains roughly **0--2 transactions per
    day**.
-   Fraud rate rises as daily transaction frequency increases and peaks
    around **7--8 transactions per day** in the EDA.
-   After this range, the observed fraud rate declines.
-   Average activity on a **normal day is about 3.4 transactions**,
    compared with approximately **5 transactions on a fraud day**.

This suggests that an unusual burst of activity can be a useful
behavioral warning signal.

------------------------------------------------------------------------

## ⚡ Transaction Interval Analysis

Transaction speed is one of the clearest fraud indicators found during
EDA.

-   Median interval for normal transactions: approximately **250
    minutes**
-   Median interval for fraudulent transactions: approximately **70
    minutes**
-   Highest observed fraud rate: approximately **2.05%** for intervals
    of **0--5 minutes**
-   Fraud rate falls to approximately **1.37%** at **15--30 minutes**
-   It decreases further to approximately **0.23%** at **3--6 hours**

> **Key behavioral signal:** fraudulent transactions tend to occur much
> faster than normal transactions.

------------------------------------------------------------------------

## 💸 Amount × Transaction Interval

The joint behavior of transaction amount and transaction speed provides
additional separation between normal and fraudulent activity.

-   Transaction amount and transaction interval show a negative
    relationship in the EDA.
-   Fraudulent observations are more concentrated in specific
    amount/time regions.
-   The highest-risk combinations occur when **larger amounts are
    transferred within very short intervals**.
-   Low amounts combined with long transaction intervals show
    considerably lower fraud rates.

------------------------------------------------------------------------

## 🎯 Customer-Level Findings

Customers with relatively short transaction histories show unusually
high observed fraud rates in the EDA.

The highest rates occur among customers with approximately **6--20 total
transactions**, while customers with larger transaction histories show
much lower rates.

This pattern suggests that account history and behavioral maturity can
provide useful contextual signals.

------------------------------------------------------------------------

# 🌳 Decision Tree Analysis

Before the final model comparison, Decision Trees were used to
investigate interpretable fraud thresholds.

### Transaction Amount

`amt` was the strongest feature in the initial encoded Decision Tree,
with an importance of approximately **0.687**.

Important amount thresholds discovered by the tree included values
around:

``` text
268.92
355.38
655.99
712.97
820.23
1,084.74
1,176.22
1,384.44
```

### Transaction Hour

The tree identified strong time splits around:

``` text
03:30
21:30
22:30
```

### Amount + Hour

Combining `amt` and `hour` revealed interpretable high-risk regions. In
particular, the tree produced fraud-class leaves for some combinations
involving:

-   transactions after approximately **21:30**
-   transaction amounts above approximately **712.97**
-   specific late-night amount ranges

This reinforces the EDA finding that **amount and transaction time
interact strongly** in fraud behavior.

------------------------------------------------------------------------

# 🤖 Machine Learning Models

Three ensemble classifiers were evaluated on the same stratified test
set:

1.  **Random Forest**
2.  **AdaBoost**
3.  **XGBoost**

Because the positive class represents only about **0.52%** of the data,
model comparison focuses especially on fraud-class Precision, Recall,
F1, and ROC-AUC.

## 📈 Model Performance

  ---------------------------------------------------------------------------
  Model          Accuracy    Precision       Recall           F1      ROC-AUC
  ---------- ------------ ------------ ------------ ------------ ------------
  **Random     **0.9969**   **0.7869**   **0.5350**   **0.6370**       0.9736
  Forest**                                                       

  AdaBoost         0.9948       0.0000       0.0000       0.0000       0.9708

  XGBoost          0.9967       0.7779       0.4927       0.6033   **0.9892**
  ---------------------------------------------------------------------------

### Random Forest Confusion Matrix

``` text
                 Predicted
               Normal   Fraud
Actual Normal   368090     277
Actual Fraud       889    1023
```

Random Forest detected **1,023 of 1,912 fraud cases** at the default
classification threshold.

### XGBoost Confusion Matrix

``` text
                 Predicted
               Normal   Fraud
Actual Normal   368098     269
Actual Fraud       970     942
```

XGBoost produced the **highest ROC-AUC (0.9892)**, indicating strong
ranking/separation ability, while its default-threshold Recall and F1
were lower than Random Forest.

### AdaBoost

At the default threshold, AdaBoost predicted no fraud cases:

``` text
True Negatives : 368367
False Positives: 0
False Negatives: 1912
True Positives : 0
```

Its high Accuracy is therefore misleading in this highly imbalanced
setting.

------------------------------------------------------------------------

# 🎚️ Threshold Tuning

Classification thresholds from **0.05 to 0.50**, in steps of **0.05**,
were evaluated for all three models.

For every threshold, the following were calculated:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   True Negatives
-   False Positives
-   False Negatives
-   True Positives

The notebook then selects the best threshold for each model according to
**F1-score**.

> The saved notebook output contains the threshold-analysis workflow and
> plots, but the rendered Styler table does not preserve the numeric
> best-threshold rows. For that reason, no unrecorded threshold values
> are claimed here.

------------------------------------------------------------------------

# 🧠 Feature Importance

Average feature importance across Random Forest, AdaBoost, and XGBoost:

    Rank Feature                     Mean Importance
  ------ ------------------------- -----------------
       1 **`amt`**                      **0.585991**
       2 **`hour`**                     **0.283232**
       3 **`time_diff_minutes`**        **0.058923**
       4 `city_pop`                         0.016926
       5 `lat`                              0.014184
       6 `merch_long`                       0.013838
       7 `merch_lat`                        0.013596
       8 `long`                             0.013311

The three dominant predictive signals are therefore:

**Transaction Amount → Transaction Hour → Time Between Transactions**

This result is consistent with the patterns observed independently
during EDA.

------------------------------------------------------------------------

# 🔑 Key Findings

  -----------------------------------------------------------------------
  Signal                              Finding
  ----------------------------------- -----------------------------------
  🌙 **Risky hours**                  Fraud is concentrated around
                                      late-night / early-morning hours,
                                      especially approximately
                                      21:00--03:00.

  ⚡ **Transaction velocity**         Very short intervals between
                                      transactions are strongly
                                      associated with fraud.

  💰 **Amount**                       Transaction amount is the most
                                      influential feature across the
                                      final models.

  📅 **Daily behavior**               Fraud days contain more
                                      transactions on average than normal
                                      days.

  👤 **Customer history**             Customers with relatively few
                                      historical transactions show high
                                      observed fraud rates in the EDA.

  🔥 **Combined risk**                Higher amounts combined with very
                                      short transaction intervals form
                                      especially suspicious regions.

  🌲 **Random Forest**                Highest default-threshold F1 and
                                      Recall among the three final
                                      models.

  🚀 **XGBoost**                      Highest ROC-AUC, reaching
                                      **0.9892**.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 💬 Results & Discussion

The analysis shows that fraudulent transactions differ meaningfully from
normal behavior.

Fraud is especially associated with:

-   unusual transaction timing,
-   short gaps between consecutive transactions,
-   increased transaction frequency,
-   transaction amount,
-   and interactions between amount and time.

The EDA and model-based feature importance independently point to the
same core signals: **amount, hour, and transaction velocity**.

At the default threshold, Random Forest provides the strongest F1/Recall
trade-off among the evaluated models, while XGBoost achieves the
strongest ROC-AUC. This distinction is important in fraud detection
because the operational threshold should ultimately depend on the
relative cost of **false negatives** versus **false positives**.

The discovered behavioral patterns can also support interpretable
**rule-based fraud screening** or serve as engineered inputs to more
advanced fraud-detection systems.

------------------------------------------------------------------------

# 🧰 Tech Stack

``` text
Python
Pandas
NumPy
Matplotlib
scikit-learn
XGBoost
Jupyter Notebook
```

------------------------------------------------------------------------

# 🚀 Running the Project

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd <your-repository>
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

Activate it on Windows:

``` bash
.venv\Scripts\activate
```

or macOS/Linux:

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

### 4. Prepare the dataset

Place the processed dataset used by the notebook as:

``` text
combined.csv
```

in the notebook's working directory.

### 5. Run the notebook

``` bash
jupyter notebook machine_learning.ipynb
```

------------------------------------------------------------------------

# 📁 Suggested Repository Structure

``` text
fraud-detection/
│
├── machine_learning.ipynb
├── combined.csv
├── README.md
├── requirements.txt
│
└── assets/
    └── plots/
```

> Large datasets should generally not be committed directly to Git. If
> `combined.csv` is large, provide a dataset source/download instruction
> instead.

------------------------------------------------------------------------

# 🔮 Possible Improvements

Future iterations can extend this work through:

-   cost-sensitive threshold optimization,
-   Precision--Recall curve analysis,
-   cross-validation designed for imbalanced data,
-   probability calibration,
-   additional customer-level rolling-window features,
-   transaction velocity features over multiple time windows,
-   anomaly-detection approaches,
-   model explainability with SHAP,
-   and evaluation using business-specific false-positive /
    false-negative costs.

------------------------------------------------------------------------

## 🏁 Conclusion

This project demonstrates that fraud detection benefits strongly from
combining **behavioral EDA, temporal feature engineering, interpretable
tree analysis, and ensemble machine learning**.

The most consistent fraud indicators discovered throughout the project
are:

> **Transaction amount, transaction hour, and the time interval between
> consecutive transactions.**

These features provide a strong foundation for both
machine-learning-based fraud detection and interpretable
fraud-monitoring rules.

------------------------------------------------------------------------

### ⭐ If you find this project useful, consider giving the repository a star.
