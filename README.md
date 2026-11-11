# PySpark E-Commerce Machine Learning & Market Segmentation Pipeline

## Overview
Distributed Big Data pipeline developed in **PySpark (Spark MLlib)** to process millions of e-commerce interaction logs (2019-Oct dataset). The project addresses two core business objectives: predicting customer purchase behavior under severe class imbalance (~98% views vs. 2% purchases) using supervised classification, and discovering natural customer economic profiles via unsupervised clustering.

---

## Key Features & Achievements
* **Architected a distributed data preparation pipeline** in PySpark, processing over 42M raw records and executing stratified sampling (`.sampleBy()`) to isolate 315K+ high-signal events.
* **Mitigated extreme class imbalance** by assigning cost-sensitive inverse class weights (`weightCol`), forcing the classifier to penalize minority-class prediction errors.
* **Built a cross-validated Random Forest pipeline** using `StringIndexer`, `VectorAssembler`, and `CrossValidator` (3-Fold with `numTrees` up to 20 and `maxDepth` up to 10), achieving an **ROC-AUC of 0.6738**.
* **Engineered high-cardinality categorical features** by tuning `maxBins=1500` to handle 1,100+ unique brands without driver memory collapse.
* **Segmented market profiles using K-Means Clustering**, leveraging `StandardScaler` feature normalization to achieve a **Silhouette Coefficient of 0.7112** across 3 distinct economic tiers.

---

## Technical Stack
* **Core Engine:** PySpark v4.0 (Spark Session, PySpark SQL, Spark MLlib)
* **Supervised Machine Learning:** Random Forest Classifier, BinaryClassificationEvaluator (ROC-AUC), ParamGridBuilder, CrossValidator (3-Fold)
* **Unsupervised Machine Learning:** K-Means Clustering, ClusteringEvaluator (Silhouette Score), StandardScaler
* **Feature Engineering:** StringIndexer, VectorAssembler, Stratified Sampling (`sampleBy`), Inverse Class Weighting
* **Environment:** Google Colab / PySpark Distributed Environment

---

## Pipeline Architecture & Methodology

### 1. Data Ingestion & Stratified Sampling
Raw clickstream data contains severe class skewness. A controlled stratified sample was extracted across target labels (`compro = 1.0` for purchases, `0.0` for views):
* **Non-Purchasers (`0.0`):** 1% sampling rate ($241,791$ records) capturing baseline browsing behavior.
* **Purchasers (`1.0`):** 10% sampling rate capturing high-intent conversion events.
* **Dataset Size:** 315,392 clean, non-null interaction records.

### 2. Supervised Classification (Random Forest)
* **Grid Search Tuning:** Evaluated combinations of `numTrees` $[10, 20]$ and `maxDepth` $[5, 10]$.
* **Best Model Configuration:** `numTrees = 20`, `maxDepth = 10`.
* **Evaluation Metric:** **ROC-AUC = 0.6738**, confirming strong predictive signal over random baselines when identifying high-value brand/price purchase triggers.

### 3. Unsupervised Market Segmentation (K-Means)
Evaluated cluster separation geometry using `ClusteringEvaluator` ($k=3$, `seed=42`), identifying three actionable business personas:

| Cluster | Persona / Segment | Avg Price (USD) | Conversion Rate | Event Volume | Strategic Business Insight |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Cluster 0** | **Mass Market / Mid-Tier** | **$218.90** | **17.91%** | 241,791 | High-volume core revenue driver of the platform. |
| **Cluster 1** | **Premium Buyers / High-End** | **$1,059.21** | **19.03%** | 53,805 | Highest conversion intent; prime target for premium loyalty programs. |
| **Cluster 2** | **Bargain Hunters / Low-End** | **$118.64** | **7.93%** | 19,796 | Low purchasing intent; target for discount incentives and cart recovery. |

---

## Diagnostic Findings & Future Iterations
* **High-Cardinality Handling:** Increasing `maxBins` to $1500$ was essential for PySpark tree algorithms to split on non-numeric brand metadata without categorical memory errors.
* **Feature Enhancement:** While static brand and price features provide strong baseline segmentation, future iterations will incorporate sequential user session features (e.g., time-on-page, cart additions) to further boost classification recall.
