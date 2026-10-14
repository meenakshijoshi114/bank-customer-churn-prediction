🏦 Bank Customer Churn Prediction

A machine learning project that predicts which bank customers are likely to leave, with an interactive Streamlit app where anyone can enter a customer's details and get a churn risk score.

Problem

Keeping an existing customer costs far less than winning a new one. This project builds a model that flags customers at risk of leaving, so a bank can offer them retention deals before they go.

Dataset
Bank customer churn dataset from Kaggle (Churn_Modelling.csv)
10,000 customers, 14 columns (credit score, country, gender, age, tenure, balance, number of products, card and activity status, salary)
Target: Exited (1 = customer left)
Classes are imbalanced: about 20% churn vs 80% stayed
No missing values

The CSV is not included in this repo. Download it from Kaggle and place it in the project folder to retrain.

Approach
Cleaning: dropped ID columns (RowNumber, CustomerId, Surname) and one-hot encoded Geography and Gender
Exploration: churn rate by age, products, activity, country, gender, balance, tenure and credit score
Modeling: Logistic Regression, Random Forest and Gradient Boosting, using a stratified 80/20 split and class-weighted or boosted models for the imbalance
Evaluation: recall, precision, F1 and ROC-AUC on the churn class, because accuracy is misleading with 80/20 classes
Threshold tuning: lowered the decision threshold from 0.5 to 0.3 to catch more churners
Deployment: saved the final model and built a Streamlit app
Key findings
Finding	Evidence
Age is the strongest signal	Churn peaks at ages 50-60 (~56%) vs a ~20% average
Product count has a U-shape	2 products: ~8% churn. 1 product: ~28%. 3-4 products: ~83-100% (small groups)
Inactive members leave more	~27% churn vs ~14% for active members
Germany stands out	~32% churn vs ~16% for France and Spain
Women churn more than men	~25% vs ~16%
Tenure and credit card ownership barely matter	Flat churn rates across groups

Feature importance from the final model agrees: Age, NumOfProducts, IsActiveMember, Balance and Geography_Germany rank highest, and HasCrCard and Tenure are near zero.

Model results
Model	ROC-AUC
Logistic Regression	0.777
Random Forest	0.855 (5-fold CV: 0.856 ± 0.004)
Gradient Boosting (final)	0.871 (5-fold CV: 0.866 ± 0.002)

Final model at threshold 0.3 (test set of 2,000 customers, 407 real churners):

Metric (churners)	Value
Recall	0.65 (266 of 407 caught)
Precision	0.63
F1	0.64
False alarms	153 loyal customers flagged

Lowering the threshold from 0.5 raised recall from 49% to 65%, a good trade for retention, where missing a churner usually costs more than contacting someone who would have stayed.

Business recommendations
Target retention offers at customers aged 40-60, especially inactive members
Investigate why customers with 3-4 products almost always leave (bundle or pricing issue?)
Look into the German market, where churn is about double
Run reactivation campaigns for inactive members before they leave
Run it locally
bash
git clone https://github.com/meenakshijoshi114/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction
pip install -r requirements.txt
python -m streamlit run app.py

The trained model (model.joblib) is included, so the app runs without retraining. To retrain, put Churn_Modelling.csv in the folder and run:

bash
python train_model.py Churn_Modelling.csv
Project structure
├── app.py                    # Streamlit app
├── train_model.py            # trains and saves the final model
├── model.joblib              # saved Gradient Boosting model
├── notebook652a8ee536.ipynb  # EDA and model comparison
├── requirements.txt
└── README.md
Limitations
The 0.3 threshold was chosen by looking at the test set, so precision and recall at that threshold are slightly optimistic
The data is a single snapshot with no time dimension, so findings show correlation, not cause
Customers with 3-4 products are rare, so those churn rates are less certain
The final model was trained on all 10,000 rows, so app probabilities may differ slightly from the notebook test results
Tools

Python, pandas, scikit-learn, matplotlib, Streamlit, Kaggle Notebooks

