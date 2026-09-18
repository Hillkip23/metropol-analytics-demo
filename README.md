Metropol Analytics Dashboard
A Streamlit prototype for integrated credit-risk intelligence, portfolio monitoring, collections prioritization, data-quality controls, policy retrieval, alternative-data scenarios, and graph-based fraud/identity review.
Prototype and synthetic-data notice: This application uses synthetic demonstration records and illustrative assumptions. It is not a production credit-decisioning, consumer-scoring, collections-decision, or fraud-adjudication system.
Included modules
1.	Executive Overview — A concise view of the end-to-end portfolio and operating-risk story.
2.	Portfolio Stress Testing — Synthetic FX returns are analysed with GARCH(1,1), peaks-over-threshold Extreme Value Theory, Student-t Monte Carlo simulation, and an illustrative Kupiec backtest.
3.	Collections Intelligence — Demonstration XGBoost and Logistic Regression models create a prioritization queue, report holdout AUC, and show borrower-level SHAP drivers.
4.	Data Quality Controls — Synthetic credit-information-provider submissions are tested with deterministic rules and Isolation Forest anomaly detection.
5.	Crystobol Assistant — A transparent TF-IDF retrieval demonstration for policy and operational guidance. It does not use an LLM or connect to a live document repository.
6.	Alternative-Data Simulation — An illustrative score-impact scenario based on mobile-money history, utility/rent payment behaviour, and income-pattern verification.
7.	Fraud & Identity Networks — A NetworkX graph visualizes synthetic shared-identifier clusters for analyst review.
8.	Methodology & Governance — A record of real analytical elements, prototype limitations, and production-readiness requirements.
Project structure
metropol-analytics-dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── synthetic_kenya_credit_data.csv
└── .streamlit/
    └── config.toml

Installation
Windows PowerShell
mkdir metropol-analytics-dashboard
cd metropol-analytics-dashboard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py

If PowerShell blocks activation, run this once in a PowerShell window:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

macOS or Linux
mkdir metropol-analytics-dashboard
cd metropol-analytics-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py

The application should open at http://localhost:8501.
Data file
Place the supplied CSV in exactly this location:
data/synthetic_kenya_credit_data.csv

The app can generate fallback synthetic data if the supplied data file is absent or cannot be read. However, you should retain the expected file structure for repeatable local and cloud deployment.
Streamlit Community Cloud deployment
1.	Create a GitHub repository, for example metropol-analytics-dashboard.
2.	Upload app.py, requirements.txt, README.md, .gitignore, the data folder, and the .streamlit folder.
3.	Commit and push all files to GitHub.
4.	In Streamlit Community Cloud, select Create app.
5.	Select your GitHub repository and branch.
6.	Set the main file path to app.py.
7.	Deploy.
8.	Open the deployment logs if the app does not load; the first reported Python exception is normally the actionable one.
Git commands
git init
git add .
git commit -m "Initial Metropol Analytics prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/metropol-analytics-dashboard.git
git push -u origin main

Replace YOUR-USERNAME with your GitHub username.
Production limitations and safeguards
This prototype is designed for research, demonstrations, and concept discussions. A production implementation should include:
•	Approved purpose, accountable business ownership, and model-risk governance.
•	Data contracts, lineage, privacy controls, security, retention policies, and access controls.
•	Appropriate lawful basis and consumer communication for any personal or alternative data.
•	Independent model validation, temporal testing, calibration checks, stability monitoring, and fairness assessment.
•	Explainability, documented operating thresholds, human review, overrides, escalation procedures, and audit trails.
•	Monitoring for data quality, model drift, performance, fairness, security events, and operational incidents.
•	Consumer dispute, correction, support, and adverse-action processes where applicable.
Disclaimer
Nothing in this prototype is financial, legal, regulatory, credit, or consumer-decision advice. Do not use any output as the sole basis for an action affecting a person or customer.
