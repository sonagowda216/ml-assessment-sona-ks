# Part B: Business Case Analysis
## Scenario: Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation (8 marks)

### (a) — ML Problem Formulation (3 marks)

**Target variable:** `items_sold` — the number of items sold at a given store in a given month.

**Candidate input features:**
- Store-level attributes: `store_size`, `location_type` (urban/semi-urban/rural), `competition_density`
- Temporal features: `month`, `year`, `is_weekend`, `is_festival`
- Customer-level features: `monthly_footfall`, demographic proxies
- Promotion deployed: `promotion_type` (Flat Discount, BOGO, Free Gift, Category-Specific, Loyalty Points)

**Type of ML problem:** This is a **supervised regression** problem. We have labelled historical data (past store-month records with known items sold), and we want to predict a continuous numeric output (items sold) for future store-month combinations.

If the business frames it as "which promotion to deploy" rather than "how many items will we sell," it could also be cast as a **multi-class classification** problem where the class is the promotion that maximises items sold. However, regression is the more flexible starting point since it allows you to score all five promotions per store and pick the highest-predicted one.

---

### (b) — Items Sold vs Revenue as Target Variable (3 marks)

Using **total sales revenue** as the target is misleading for a few reasons:

1. **Price distortion:** A Flat Discount reduces the unit price, so revenue can drop even if more items are sold. The marketing team would falsely conclude the promotion "failed" when it actually drove higher volume.

2. **Inconsistent margin accounting:** BOGO gives away one item free — this halves revenue per pair sold, making it look worse than a Loyalty Points campaign on a revenue basis, even if BOGO moves the most physical inventory.

3. **The actual business goal is volume maximisation** — the brief explicitly states "maximise the number of items sold," not revenue. Using a proxy metric that doesn't align with the objective corrupts the entire modelling process.

**Broader principle — target variable alignment:**
In real-world ML, target variable selection is arguably the most important design choice. The target must directly measure what the business is optimising for. If there's a mismatch (e.g., using proxy metrics like revenue when the goal is volume), even a perfectly trained model will drive the wrong decisions. Always trace back to the business objective before defining the target.

---

### (c) — Global vs Location-Stratified Modelling (2 marks)

Running a single global model across all 50 stores is problematic because it assumes the same promotion-sales relationship holds regardless of location. However, a BOGO promotion may work brilliantly in urban stores (dense competition, deal-hunting shoppers) and fall flat in rural ones (lower footfall, different customer demographics).

**Alternative strategy — Location-stratified models:**
Train **three separate models** — one each for Urban, Semi-Urban, and Rural store clusters. This captures location-specific promotion sensitivity while keeping the number of models manageable.

A further refinement would be a **hierarchical/mixed-effects model** that shares information across stores (useful for smaller stores with fewer months of data) while still allowing store-level intercepts and promotion coefficients to vary.

---

## B2. Data and EDA Strategy (10 marks)

### (a) — Joining Tables and Data Grain (4 marks)

The raw data arrives in four tables:
- **Transactions** — one row per transaction: `store_id`, `transaction_date`, `promotion_applied`, `items_sold`
- **Store attributes** — one row per store: `store_id`, `store_size`, `location_type`, `monthly_footfall`, `competition_density`
- **Promotion details** — one row per promotion type: `promotion_type`, `discount_rate`, `category_applicable`
- **Calendar** — one row per date: `date`, `is_weekend`, `is_festival`

**Join strategy:**
1. Aggregate transactions to **store × month grain** — sum `items_sold`, note which `promotion_type` was run that month.
2. Left join store attributes on `store_id`.
3. Left join calendar on `month` / aggregated date fields (since promotions run all month, join on year-month).
4. Left join promotion details on `promotion_type`.

**Final grain:** One row = one store × one month.  
Each row captures: which store, which month, which promotion was deployed, store characteristics, calendar flags, and total items sold that month.

**Aggregations performed:**
- Sum `items_sold` per store-month
- Mode or most frequent `promotion_type` per store-month (if multiple in one month)
- Weekend/festival flags: fraction of days in the month that were weekends/festivals (or a simple binary for "did the month contain a festival")

---

### (b) — EDA Before Modelling (4 marks)

Four analyses / charts and what to look for:

1. **Items sold by promotion type (box plot):**  
   Compare the distribution of `items_sold` across all five promotion types. Look for which promotion has a higher median and tighter distribution — outliers here may indicate interaction with store size or location. Findings directly inform feature engineering (e.g., creating interaction terms like `promo_type × location_type`).

2. **Monthly trend of items sold (line chart over time):**  
   Check for seasonality — spikes in certain months (back-to-school, festive seasons) indicate that `month` and `is_festival` will be important features. Also reveals if there's a general upward trend that needs detrending.

3. **Correlation heatmap of numerical features vs items_sold:**  
   Identify which numeric features (`store_size`, `competition_density`, `footfall`) correlate most strongly with the target. Highly collinear pairs (e.g., if `footfall` and `store_size` are correlated) may need to be addressed via feature selection or PCA to avoid multicollinearity in linear models.

4. **Promotion effectiveness by location type (grouped bar chart):**  
   For each location type (Urban/Semi-Urban/Rural), show the average items sold per promotion. This directly validates or refutes the need for location-stratified models — if the bar heights differ significantly across locations, a global model is insufficient.

---

### (c) — Class Imbalance: 80% Without Promotion (2 marks)

If 80% of transactions occurred without any promotion, the model may learn that "no promotion" is the default and fail to distinguish between the five promotion types that occupy only 20% of the data.

**Effects:**
- Feature coefficients for promotion-related features will be underfit
- The model predicts close to the "no promotion" average for all inputs
- Evaluation metrics (RMSE) may look acceptable because the majority of rows are well-predicted, masking poor performance on promoted periods

**Steps to address:**
1. **Stratified sampling during train-test split** — ensure promotion months are proportionally represented in both sets.
2. **Oversample promotion rows** using techniques like SMOTE (if reformulated as classification) or simply duplicate promotion rows in the training set.
3. **Separate model for promoted vs non-promoted periods** — train a "promotion effect model" on rows with promotions, then combine predictions.
4. **Use promotion-month-weighted loss** — assign higher sample weights to promotion records so the model penalises errors on them more.

---

## B3. Model Evaluation and Deployment (12 marks)

### (a) — Train-Test Split Strategy and Evaluation Metrics (4 marks)

**Train-test split setup:**  
With 3 years × 50 stores = ~1800 store-month records, a **temporal holdout** is correct. Train on the first 28–30 months, test on the final 6 months. This mirrors real usage: the model is always predicting future months it has not seen.

**Why random split is inappropriate:**  
A random split would scatter future months into the training set. The model would learn from December 2024 data and "predict" October 2023 — this is leakage. Real-world deployment only ever predicts forward in time, so evaluation must reflect that.

**Evaluation metrics:**

| Metric | Formula | Business Interpretation |
|--------|---------|------------------------|
| **RMSE** | √(mean((ŷ−y)²)) | Penalises large errors heavily; useful to avoid badly wrong recommendations for high-footfall stores |
| **MAE** | mean(\|ŷ−y\|) | "On average, our prediction is off by X items." Easier to communicate to non-technical stakeholders |
| **R²** | 1 − SS_res/SS_tot | Proportion of variance explained; useful to benchmark against a naïve mean-prediction baseline |
| **Promotion Ranking Accuracy** | % of store-months where the model correctly identifies the best promotion | Most directly aligned to the business goal of choosing the right promotion |

---

### (b) — Feature Importance and Explaining Different Recommendations (4 marks)

The model recommends **Loyalty Points Bonus in December** and **Flat Discount in March** for Store 12. These are different months, so the model is correctly responding to month-specific context — not making an error.

**How to investigate using feature importance:**
1. Extract SHAP values (or standard RF feature importances) for Store 12's December and March records separately.
2. For December, identify which features pushed the prediction toward Loyalty Points — likely `is_festival=1` (December holidays), high `monthly_footfall`, and the historical response of Store 12 to loyalty promotions during high-footfall months.
3. For March, check which features drive Flat Discount recommendation — possibly lower footfall, post-holiday budget-consciousness, and higher `competition_density` (competitors doing clearance sales).

**How to communicate to the marketing team:**
> "In December, Store 12 experiences heavy footfall from loyal customers — these shoppers respond better to Loyalty Points because they plan to return anyway and the accumulated points increase retention. In March, footfall drops and price sensitivity rises, so a straightforward Flat Discount attracts more footfall-driven, occasional shoppers. The model is correctly recognising that the same promotion doesn't fit all months for the same store."

This framing builds trust in the model by showing it is capturing real business dynamics rather than outputting black-box recommendations.

---

### (c) — End-to-End Deployment Process (4 marks)

**Saving the model:**
```python
import joblib
joblib.dump(pipeline, 'promotion_recommender_v1.pkl')
```
Store model artifacts (pipeline, scaler, encoder mappings) in a versioned object store (e.g., S3 or Azure Blob Storage) alongside a model card documenting training data date range, feature list, and evaluation metrics.

**Monthly inference workflow:**
1. At the start of each month, the data engineering pipeline automatically:
   - Pulls updated store attributes, prior month's transaction data, and the upcoming month's calendar (festivals, weekends)
   - Joins tables to produce a 50-row feature matrix (one row per store)
2. Load the saved model pipeline.
3. Score all 50 stores × 5 promotion types (generate 250 predictions).
4. For each store, select the promotion type with the highest predicted `items_sold`.
5. Output a recommendation table: `store_id → recommended_promotion_type → predicted_items_sold`.
6. Deliver to the marketing team via a dashboard (e.g., Tableau, Power BI, or a simple email report).

**Monitoring for model degradation:**
- **Prediction drift monitoring:** Track the distribution of predicted `items_sold` each month. If predictions shift significantly without corresponding real-world changes, it signals model staleness.
- **Actual vs predicted tracking:** After each month, compare the recommended promotion's actual result vs the model's prediction. Log RMSE month-over-month.
- **Data drift detection:** Use statistical tests (e.g., Kolmogorov-Smirnov, Population Stability Index) on input features to detect if store characteristics, footfall patterns, or promotion mix have changed.
- **Retraining trigger:** If rolling 3-month RMSE exceeds 1.5× the baseline evaluation RMSE, or if a data drift alert fires, trigger a retraining job using the full updated historical dataset.
- **A/B testing integration:** Periodically run controlled experiments — deploy the model's recommendation to 40 stores and a challenger strategy (e.g., human planner's pick) to 10, compare outcomes, and use results to guide model updates.
