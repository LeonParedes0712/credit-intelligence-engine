\# Credit Intelligence Engine



Credit Intelligence Engine is an interpretable credit risk decision system built with machine learning, business rules, credit scoring, explainability, and financial risk simulation.



The goal of this project is not only to train a default prediction model, but to build an end-to-end credit decision engine that converts model outputs into actionable lending decisions.



\## Project Overview



This project uses the Home Credit Default Risk dataset to predict the probability that a credit applicant will default.



The engine performs the following steps:



1\. Predicts probability of default using machine learning.

2\. Converts default probability into a 300–850 credit score.

3\. Assigns risk tiers.

4\. Applies business rules to adjust decisions.

5\. Explains model behavior using SHAP.

6\. Simulates approval thresholds.

7\. Estimates expected loss and expected profit.

8\. Provides an interactive Streamlit app for applicant simulation.



\## Dataset



Dataset used:



Home Credit Default Risk - Kaggle



Main file used in the current version:



```text

data/application\_train.csv



Target variable:



TARGET



Where:



0 = no default

1 = default



The dataset is not included in this repository because of its size. The data/ folder is ignored by Git.



Repository Structure

credit-intelligence-engine/

│

├── app/

│   └── streamlit\_app.py

│

├── data/

│   └── application\_train.csv          # Not tracked by Git

│

├── notebooks/

│   ├── 01\_eda\_baseline.ipynb

│   ├── 02\_xgboost\_model.ipynb

│   ├── 03\_credit\_score\_engine.ipynb

│   ├── 04\_business\_rules\_engine.ipynb

│   ├── 05\_shap\_explainability.ipynb

│   ├── 06\_model\_calibration\_thresholds.ipynb

│   └── 07\_expected\_loss\_profit.ipynb

│

├── outputs/

│   ├── xgboost\_model.pkl              # Not tracked by Git

│   └── generated result files          # Not tracked by Git

│

├── src/

├── requirements.txt

├── .gitignore

└── README.md

Methodology

1\. Exploratory Data Analysis and Baseline Model



The first notebook explores the application dataset, checks missing values, reviews target imbalance, and trains a Logistic Regression baseline.



Baseline model:



Logistic Regression + preprocessing pipeline



Baseline ROC-AUC:



0.6238

2\. XGBoost Credit Risk Model



A stronger tree-based model was trained using XGBoost.



The model uses:



Median imputation for numerical variables

Most frequent imputation for categorical variables

One-hot encoding

XGBoost classifier

Class imbalance handling with scale\_pos\_weight



XGBoost ROC-AUC:



0.7598



This significantly improved performance compared to the Logistic Regression baseline.



3\. Credit Score Engine



Predicted default probabilities are converted into a credit score from 300 to 850.



Higher default probability produces a lower credit score.



Example logic:



Low PD  -> Higher score

High PD -> Lower score



Risk tiers:



750+     Very Low Risk

700-749  Low Risk

650-699  Medium Risk

600-649  High Risk

<600     Very High Risk

4\. Business Rules Layer



A business rules layer was added on top of the model decision.



Rules include:



Very low credit score

High predicted default probability

High credit amount relative to income

High annuity burden relative to income

Low external risk score

Short employment history



This allows the engine to produce both a final decision and business-readable reasons.



Decision outputs:



Approve

Manual Review

Reject

5\. SHAP Explainability



SHAP was used to explain the XGBoost model.



The project includes:



Global feature importance

SHAP summary plot

Individual applicant explanation with waterfall plot



This improves interpretability by showing which variables increase or decrease predicted default risk.



6\. Calibration and Threshold Simulation



The model was evaluated using:



ROC-AUC

Brier score

Calibration curve

Threshold simulation



A recommended initial policy was selected:



Approve if PD <= 30%

Manual Review if 30% < PD < 50%

Reject if PD >= 50%



This policy produced approximately:



Approval rate: 34.6%

Approved default rate: 2.2%

7\. Expected Loss and Profit Simulation



A financial risk layer was added using:



Expected Loss = PD × LGD × EAD



Where:



PD  = Probability of Default

LGD = Loss Given Default

EAD = Exposure at Default



In this project:



LGD = 45%

EAD = AMT\_CREDIT



Expected profit was estimated as:



Expected Profit = Expected Interest - Expected Loss



A sensitivity analysis was performed using different interest rate assumptions.



Streamlit App



The project includes an interactive Streamlit app that allows users to simulate a credit application.



The app outputs:



Default probability

Credit score

Risk tier

Final credit decision

Expected interest

Expected loss

Expected profit

Business rule reasons

Applicant data used by the engine



To run the app:



streamlit run app/streamlit\_app.py



On Windows, if Streamlit is installed in the Python environment:



py -m streamlit run app/streamlit\_app.py

Installation



Clone the repository:



git clone https://github.com/LeonParedes0712/credit-intelligence-engine.git

cd credit-intelligence-engine



Create a virtual environment.



On Windows:



py -m venv .venv

.venv\\Scripts\\activate

py -m pip install -r requirements.txt



On Linux/macOS:



python3 -m venv .venv

source .venv/bin/activate

python -m pip install -r requirements.txt

Required Local Files



Because large data and model artifacts are not tracked by Git, the following files must exist locally to run the full project:



data/application\_train.csv

outputs/xgboost\_model.pkl



The dataset can be downloaded from Kaggle:



Home Credit Default Risk



The model file can be generated by running:



notebooks/02\_xgboost\_model.ipynb

Key Results

Component	Result

Logistic Regression ROC-AUC	0.6238

XGBoost ROC-AUC	0.7598

Recommended approval threshold	PD <= 30%

Recommended reject threshold	PD >= 50%

Approved default rate	\~2.2%

LGD assumption	45%

Technologies Used

Python

pandas

NumPy

scikit-learn

XGBoost

SHAP

Streamlit

Plotly

Matplotlib

Joblib

Git / GitHub

Project Status



Current version includes:



EDA and baseline model

XGBoost model

Credit score conversion

Risk tiers

Business rules

SHAP explainability

Threshold simulation

Expected loss and profit simulation

Interactive Streamlit app



Possible future improvements:



Use additional Home Credit tables for feature engineering

Add probability calibration

Improve UI design

Add SHAP explanations directly inside the Streamlit app

Add model monitoring metrics

Package reusable logic into src/

