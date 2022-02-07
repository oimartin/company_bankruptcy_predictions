[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oimartin/company_bankruptcy_predictions/blob/main/Notebook_bankrupt.ipynb#scrollTo=ZEw9B5eKJqhN)
[![Python application test with Github Actions](https://github.com/oimartin/company_bankruptcy_predictions/actions/workflows/main.yml/badge.svg)](https://github.com/oimartin/company_bankruptcy_predictions/actions/workflows/main.yml)

# Company Bankruptcy Predictions
Data was collected from the 'Company Bankruptcy Prediction' on Kaggle (https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction). Predictions for company bankruptcy come from Taiwan Economic Journale (1999-2009). 95 input features, 1 output feature, Bankrupt?.

## Key updates
* Used over and under sampling with SMOTE plus cleaning with Tomek links to balance bankrupt companies and non-bankrupt companies
* Split data earlier in the data process
* Removed code to handle outliers and only use StandardScaler
* Used Random Forest Classifier, Gradient Boosted Trees Classifier, and Extra Tree Classifier models
* Updated metrics used to evaluate classification models
# EDA
## Initial discovery
The bankruptcy data did not have any missing values, and had a total of 6,819 rows of data. The output feature, Bankrupt?, is a categorical feature with responses of 1 indicating a bankrupt company and 0 indicating a non-bankrupt company. The split of bankrupt vs non-bankrupt is almost all data is associated with non-bankrupt companies.

![Bankrupt_feature](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/Data_on_Bankrupt_or_Not_Bankrupt_Companies.png?raw=true)

Using SMOTE Tomek from imblearn library, the bankrupt ratio changes to 1:1 with 6,562 observations for both bankrupt companies and non-bankrupt companies.

Additional exploration of the data shows the skewness of data features based on positive or negative skewness. 'CashFlowtoLiability' feature has a skewness of -3.13866, and 'NetValueGrowthRate' has a skewness of 23.4064. 
![Positevely_Skewed](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/positively_skewed_data.png?raw=true)
![Negatively_Skewed](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/negatively_skewed_data.png?raw=true)

## Data manipulation
The five most positively skewed and five most negatively skewed features were evaluated in the following violin/stripplots.
![Top_positevely_skewed](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/heavily_positively_skewed_violin_strip.png?raw=true)
![Top_negatively_skewed](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/negatively_skewed_data_violin_strip.png?raw=true)

For the features I had previously seperated into categorical and continous data, I then set the data type for categorical features astype('category'). Rebalancing the data around bankrupt and non-bankrupt companies emphasized some features with strong correlations to bankruptcy that had previously not been correlated in the imbalanced pre-processed data.
![Numeric_correlations](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/feature_corr.png?raw=true)
![correlations](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/top_bottom_correlation_bankrupt.png?raw=true)

# Model
## Model Prepeartion
The data was then split with a test size of 20% and a train size of 80%. A pipeline was created with the following steps: StandardScaler() for num_features (93 count). 

## Model Comparison
![comparison](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/accuracy_f1_bsl_log_loss_traintest_comparison.png?raw=true)
![comparison](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/recall_precision_roc_auc_test_train_comparison.png?raw=true)
## Logistic Regression
Both the train and test data had high accuracy scores, 0.9097 and 0.9029, respectively. 
![LR_train](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/LG_Train_roc_prec_rec_curves.png?raw=true)
![LR_test](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/LG_Test_roc_prec_rec_curves.png?raw=true)

## Linear Support Vector Classification
Again, the train and test data had high accuracy scores, 0.9106 and 0.9070, respectively. 
![LSVC_train](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/LSVC_Train_roc_prec_rec_curves.png?raw=true)
![LSVC_test](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/LSVC_Test_roc_prec_rec_curves.png?raw=true)

## Gaussian Naive Bayes
The GNB model performed worse than the LR and LSVC models with train and test scores, 0.6822 and 0.6888, respectively. 
![GNB_train](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/GNB_Train_roc_prec_rec_curves.png?raw=true)
![GNB_test](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/GNB_Test_roc_prec_rec_curves.png?raw=true)

## Random Forest Classifier
The RFC model performed well with train and test scores, 1.0 and 0.9775, respectively. 
![RFC_train](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/RFC_Train_roc_prec_rec_curves.png?raw=true)
![RFC_test](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/RFC_Test_roc_prec_rec_curves.png?raw=true)

## Gradient Boosted Tree Classifier
The GBT model performed well with train and test scores, 0.9703 and 0.9550, respectively. 
![GBT_train](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/GBT_Train_roc_prec_rec_curves.png?raw=true)
![GBT_test](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/GBT_Test_roc_prec_rec_curves.png?raw=true)


## Extra Tree Classifier
The ETC model performed well with train and test scores, 1.0 and 0.9756, respectively. 
![ETC_train](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/ETC_Train_roc_prec_rec_curves.png?raw=true)
![ETC_test](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/ETC_Test_roc_prec_rec_curves.png?raw=true)
