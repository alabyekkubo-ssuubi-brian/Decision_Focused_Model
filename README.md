📦 Decision-Focused Probabilistic Forecasting on the M5 Dataset

Hybrid Predict-Then-Optimize + Newsvendor Loss Training
MLP Baseline vs Decision-Focused MLP | Multi-Fold Evaluation | PyTorch

This repository implements a full experimental pipeline for decision-focused forecasting applied to the M5 Walmart Sales Dataset.
The goal is to show how training directly on decision loss (newsvendor cost) improves downstream operational metrics such as cost, fill-rate, and optimal order quantity selection.

🚀 Key Features

End-to-End Reproducible Notebook (Colab-ready)

Multi-fold temporal validation (3 folds)

Predict-then-Optimize baseline

Decision-Focused MLP using Newsvendor loss

Probabilistic forecasting via quantile regression

Hybrid loss:
alpha * PinballLoss + (1 − alpha) * DecisionLoss

Evaluation metrics:
RMSSE, RMSE, Quantile loss, Newsvendor cost, Fill-rate

XAI: SHAP value explanation for model interpretability

Visualization suite: fold-wise plots, error distributions, cost curves

Reproducibility section: random seeds, configs, environment, runtime instructions

📁 Project Structure
m5_decision_focused/
│
├── config.py                # Global hyperparameters & paths
├── data/
│   ├── raw/                 # Original M5 CSVs
│   ├── processed/           # Preprocessed, FE data
│
├── models/
│   ├── mlp_baseline.py      # Predict-then-optimize model
│   ├── mlp_decision.py      # Decision-focused model
│   └── losses.py            # Pinball + Newsvendor loss
│
├── utils/
│   ├── data_loader.py       # Load M5 + FE features
│   ├── metrics.py           # RMSSE, cost, fill-rate, etc.
│   ├── plot_utils.py        # All visualizations
│
├── notebooks/
│   ├── M5_Decision_Focused.ipynb     # Full training notebook
│   ├── EDA.ipynb                      # Exploratory analysis
│
├── results/
│   ├── fold1/ fold2/ fold3/           # Metrics, plots, checkpoints
│   ├── summary.csv
│
└── README.md

🧠 Method Summary
🔹 Baseline (Predict-Then-Optimize)

Train the model using Pinball loss on quantiles, then solve Newsvendor problem after prediction.

🔹 Decision-Focused Model

Directly optimizes the expected cost during training:

𝐿
=
𝛼
⋅
𝐿
𝑝
𝑖
𝑛
𝑏
𝑎
𝑙
𝑙
+
(
1
−
𝛼
)
⋅
𝐿
𝑁
𝑉
L=α⋅L
pinball
	​

+(1−α)⋅L
NV
	​


This pushes the model to care more about order decisions than pure forecast accuracy.

📊 Experimental Pipeline
1. Data Preprocessing

Load sales, prices, calendar tables

Convert to long format

Feature engineering:

Lag features

Rolling means

Price features

Event features

Day-of-week, month, year, etc.

Filter on a manageable subset (default: State = CA)


Folders created:

results/fold1/metrics.json
results/fold1/loss_curve.png
results/... etc

3. Evaluation

Evaluates:

RMSSE

RMSE

Newsvendor Cost

Fill-rate

Quantile Coverage

Decision error: |Q̂ − Q*|

4. Visualization

Includes:

Cost comparison (baseline vs DF)

RMSSE comparison

Quantile coverage plots

Fold-wise training curves

SHAP interpretability (top 20 features)

📈 Example Results (Summary Table)
Metric	Baseline	Decision-Focused	Δ Improvement
RMSSE	6.963	0.933	–86.6%
Avg Cost	2.34	1.01	–56.8%
Fill-Rate	0.71	0.89	+18%
Decision Error	28.5	9.2	–67%

⚠️ Note: The RMSSE anomaly (Fold 2) is under investigation. Could be a preprocessing or scaling bug.

🧪 Reproducibility
Environment
Python 3.12
PyTorch 2.x
pandas, numpy, matplotlib, seaborn
scikit-learn
shap

Set seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

Download Data

Place M5 files into:

data/raw/

Run Notebook End-to-End

Open:

notebooks/M5_Decision_Focused.ipynb

🧩 Explainability (SHAP)

To generate SHAP plots:

import shap
explainer = shap.DeepExplainer(model, sample_batch)
shap_values = explainer.shap_values(sample_batch)
shap.summary_plot(shap_values, sample_batch)


Output:

Feature importance summary

Feature contribution for individual decisions

Helps justify decision-focused behaviour

🛠️ Troubleshooting
RMSSE extremely high?

Check scaling (should be item-wise)

Check sales_fe overwrite bug (common)

Confirm y_true.shift(1) does not generate zeros

Validation cost unstable?

Increase alpha toward 0.4

Reduce learning rate

Clip gradients

Fold-wise inconsistency?

Ensure date splits align with M5 competition structure

📚 References

Elmachtoub & Grigas, “Smart Predict-Then-Optimize,” Management Science, 2022

Koenker & Bassett, “Regression Quantiles,” Econometrica, 1978

Gneiting & Raftery, “Strictly Proper Scoring Rules,” JASA, 2007

Makridakis et al., “The M5 Competition,” 2022

Lim et al., “Temporal Fusion Transformers,” 2021

Salinas et al., “DeepAR,” 2019

📝 License

MIT License

🙌 Acknowledgements

This project is part of research on decision-focused learning and reproducible forecasting methodologies.New File for Project Initialisation 
