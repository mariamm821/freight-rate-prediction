# Spotter Freight Rate Prediction
Machine learning solution for the Spotter Freight Rate Prediction Challenge.
## Objective
The goal of this project is to predict the posted freight rate for each load.
The labeled development dataset (`data/train_test.csv`) was used to explore the data, perform feature engineering, train the model, and validate its performance.
The trained model was then used to generate predictions for:
- `data/validation.csv`
- `data/december_chart_inputs.csv`
## Approach
The solution includes:
- Data exploration and quality checks
- Time-based train/validation split
- Feature engineering from date, weight, distance, pickup, delivery, and equipment
- Handling of categorical features
- CatBoost regression models
- Validation using MAE, RMSE, MAPE, and R²
- Final training using all labeled development data
- Prediction generation for the 12,000 validation loads
- December fixed-scenario predictions
- Validation using the provided `score.py`
## Validation Strategy
Because the dataset contains dates from 2025, a time-based validation strategy was used instead of a random split.
The data was divided as follows:
- **Training:** January–September 2025
- **Validation:** October 2025
This allows the model to be evaluated on a later time period than the data used for training.
### Validation Results
| Metric | Result |
|---|---:|
| MAE | $114.11 |
| RMSE | $646.12 |
| MAPE | 5.51% |
| R² | 0.821 |
The validation set contained 4,853 October loads, while 43,147 January–September loads were used for training.
## Feature Engineering
The model uses information available in the load data, including:
- Pickup location
- Delivery location
- Equipment type
- Distance
- Weight
- Weight-missing indicator
- Date-related features
Date features include calendar and cyclic representations to allow the model to capture time-related patterns.
The following identifiers and fields that were not used as predictive features were excluded from modeling where appropriate.
## Model
The final solution uses an ensemble of two CatBoost regression models.
Two target formulations were used:
1. Predicting `log1p(posted_rate)`
2. Predicting the rate normalized by distance (`posted_rate / distance`)
The predictions from the two models are combined to produce the final freight-rate prediction.
CatBoost was selected because it can handle categorical variables directly while also supporting nonlinear relationships between the input features and freight rates.
## Installation
Install the required dependencies with:
```bash
pip install -r requirements.txt
```
The main dependencies are:
```bash
matplotlib
numpy
pandas
catboost
scikit-learn
```
## Running the Pipeline
Run the main script from the project root:
```bash
python train_predict_all.py
```
The script performs the complete workflow:
1.Loads the labeled development data.
2.Performs the January–September / October validation split.
3.Trains the models.
4.Evaluates the models on the October validation set.
5.Retrains the final model using all labeled development data.
6.Generates predictions for all 12,000 rows in validation.csv.
7.Generates predictions for the 31 December fixed-scenario rows.
8.Runs the provided scorer.
## Output Files
The pipeline generates the final validation predictions:
```bash
validation_predictions.csv
```
The file contains exactly two columns:
```bash
load_id,predicted_rate
```
The pipeline also fills the predicted_rate column in:
```bash
data/december_chart_inputs.csv
```
The provided scorer then generates:
```bash
scorer_results/candidate_december.png
```
## Repository Structure
```bash
spotter-freight-rate-prediction/
│
├── train_predict_all.py
├── requirements.txt
├── README.md
├── validation_predictions.csv
│
├── data/
│   └── december_chart_inputs.csv
│
└── scorer_results/
    └── candidate_december.png
```
## Reproducibility
A fixed random seed is used in the modeling pipeline.
To reproduce the workflow:
```bash
pip install -r requirements.txt
python train_predict_all.py
```
The provided score.py validates that:
- validation_predictions.csv contains the required 12,000 predictions.
- The required load IDs are present.
- Predicted rates are valid positive numeric values.
- The December prediction file contains the required 31 rows and dates.
- The December prediction chart is generated successfully.
 
