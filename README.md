[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oimartin/company_bankruptcy_predictions/blob/main/Notebook_bankrupt.ipynb#scrollTo=ZEw9B5eKJqhN)

# Company Bankruptcy Predictions
Data was collected from the 'Company Bankruptcy Prediction' on Kaggle (https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction). Predictions for company bankruptcy come from Taiwan Economic Journale (1999-2009). 95 input features, 1 output feature, Bankrupt?.

# EDA
## Initial discovery
The bankruptcy data did not have any missing values, and had a total of 6,819 rows of data. The output feature, Bankrupt?, is a categorical feature with responses of 1 indicating a bankrupt company and 0 indicating a non-bankrupt company. The split of bankrupt vs non-bankrupt is almost all data is associated with non-bankrupt companies.
![Bankrupt_feature](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/Data_on_Bankrupt_or_Not_Bankrupt_Companies.png?raw=true)

I next seperated all 96 features into ones with a min and max of 0 and 1. The other features represented data with other min and max values. I divided the features by min and max values in order to seperate categorical features versus continous features, but after completing some model, I realized some features with min/max of 0 and 1 are also continous. In my next iteration of processing data, I would like to re-evaluate how I organize the features. In this analysis, I organized the features into 71 being categorical and 25 being continous.
![Continous_data](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/Continuous_data_distributions.png?raw=true)

Additional exploration of the data shows the skewness of data features based on positive or negative skewness. 'Bankrupt?' feature has a skewness of 5.9539, 'FixedAssetstoAssests' feature has a skewness of 82.5772, and 'OperatingProfitGrowthRate' has a skewness of -71.689. 
![Positevely_Skewed](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/positively_skewed_data.png?raw=true)
![Negatively_Skewed](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/negatively_skewed_data.png?raw=true)

## Data manipulation
The five most positively skewed and five most negatively skewed features were evaluated in the following violin/stripplots.
![Top_positevely_skewed](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/heavily_positively_skewed_violin_strip.png?raw=true)
![Top_negatively_skewed](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/negatively_skewed_data_violin_strip.png?raw=true)

The same top skewed features were compared after removing outliers from the data. Outliers removed were determined by the IQR of individual feature multiplied by 0.25 and either added to the 75th percentile or subtracted from the 25th percentile. I wanted to remove more outliers than the original function used by MART093 in his notebook: https://www.kaggle.com/marto24/bankruptcy-detection. The resulting skew profiles for previously top skewed features:
![Prev top_positevely_skewed_post_outlier_removal](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/prev_skew_data_after_outlier_removal_pos.png?raw=true)
![Prev top_negatively_skewed_post_outlier_removal](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/prev_skew_data_after_outlier_removal_neg.png?raw=true)

For the features I had previously seperated into categorical and continous data, I then set the data type for categorical features astype('category'). I initially reset 'Bankrupt?' astype('int64), so that I can use it in a correlation heat map. Later, I convert 'Bankrupt?' back astype('category'). There were very almost no features that were strongly correlated with bankruptcy. Totaldebt/Totalnetworth had a correlation to bankruptcy of about 0.2. I think in my next evaluation of the data, I would also create a heat map of numeric features correlated to 'Bankrupt?' = 1. 
![Numeric_correlations](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/feature_corr.png?raw=true)

# Model
## Model Prepeartion
The data was then split with a test size of 20% and a train size of 80%. A pipeline was created with the following steps: StandardScaler() for num_features (24 count), SimpleImputer(strategy='mostfrequent') for categorical features (75 count).

## Logistic Regression
Both the train and test data had high accuracy scores, 0.9757 and 0.9683, respectively. 
![LR_train](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/LG_Train_roc_prec_rec_curves.png?raw=true)
![LR_test](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/LG_Test_roc_prec_rec_curves.png?raw=true)

## Linear Support Vector Classification
Again, the train and test data had high accuracy scores, 0.9749 and 0.9673, respectively. 
![LSVC_train](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/LSVC_Train_roc_prec_rec_curves.png?raw=true)
![LSVC_test](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/LSVC_Test_roc_prec_rec_curves.png?raw=true)

## Gaussian Naive Bayes
The GNB model performed much worse than the LR and LSVC models with train and test scores, 0.171 and 0.1508, respectively. 
![GNB_train](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/GNB_Train_roc_prec_rec_curves.png?raw=true)
![GNB_test](https://github.com/oimartin/company_bankruptcy_predictions/blob/main/figures/GNB_Test_roc_prec_rec_curves.png?raw=true)


