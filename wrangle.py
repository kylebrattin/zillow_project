# Imports
#~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from env import get_db_url
import os
from scipy.stats import ttest_ind, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, LassoLars
from sklearn.feature_selection import RFE


# How did I acquire the data

def get_zillow():
    '''This function imports zillow 2017 data from MySql codeup server and creates a csv
    
    argument: df
    
    returns: zillow df'''
    filename = "zillow.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        query = """
        SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, lotsizesquarefeet, fips, transactiondate, garagecarcnt, garagetotalsqft
        FROM properties_2017
        JOIN propertylandusetype USING (propertylandusetypeid)
        JOIN predictions_2017 USING (parcelid)
        WHERE propertylandusetypeid like '261';
        """
        connection = get_db_url("zillow")
        df = pd.read_sql(query, connection)
        df.to_csv(filename, index=False)
    return df

#~~~~~~~~~~~~~~~
# Prep 
#~~~~~~~~~~~~~~~
# How did I prep the data
def prep_zillow(df):
    '''takes the zillow dataframe, drops nulls, drops transactiondate, changes column names, 
    and replaces null values in garage columns

    argument: df

    returns: clean_df
    '''
    # rename columns
    df.rename(columns={'fips': 'county', 'bedroomcnt': 'bedrooms', 'garagecarcnt': 'garage_fits', 'bathroomcnt': 'bathrooms', 'garagetotalsqft': 'garage_area', 'calculatedfinishedsquarefeet': 'finished_area', 'lotsizesquarefeet': 'lot_area', 'taxvaluedollarcnt': 'home_value'}, inplace=True)
    
    # garage_fits change nan, dtype=int, fill nulls for garage_area
    df.garage_fits = df.garage_fits.fillna(0).astype(int)
    df.garage_area = df.garage_area.fillna(0)
    
    # take care of null vals for yearbuilt, finished_area, lot_area, and home_value
    df.dropna(axis=0,inplace=True)
    # change yearbuilt type to int
    df.yearbuilt = df.yearbuilt.astype(int)
    # drop transactiondate
    df = df.drop(columns='transactiondate')
    # start with county, change unique values
    df.county = df.county.map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    # change bedroom dtype to int and drop values 0 and everything above 8
    df.bedrooms = df.bedrooms.astype(int)
    df.drop(df[df['bedrooms'] > 8].index, inplace=True)
    df.drop(df[df['bedrooms'] == 0].index, inplace=True)
    # drops the homes with no bathrooms
    df.drop(df[df['bathrooms'] < .5].index, inplace=True)
    # drops the bottom 1% and the top 1% to deal with outliers
    bottom_perc = df.home_value.quantile(.01)
    top_perc = df.home_value.quantile(.99)
    df = df[(df.home_value > bottom_perc) & (df.home_value < top_perc)]

    return df


#~~~~~~~~~~~~~~~
# Data Checker
#~~~~~~~~~~~~~~~
def check_columns(df):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    the number of null values in the column, and the data type of the column.
    The resulting dataframe is sorted by the 'Number of Unique Values' column in ascending order.
​
    Args:
    - df: pandas dataframe
​
    Returns:
    - pandas dataframe
    """
    data = []
    # Loop through each column in the dataframe
    for column in df.columns:
        # Append the column name, number of unique values, unique values, number of null values, and data type to the data list
        data.append(
            [
                column,
                df[column].nunique(),
                df[column].unique(),
                df[column].isna().sum(),
                df[column].isna().mean(),
                df[column].dtype
            ]
        )
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', 'Number of Null Values', and 'dtype'
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return pd.DataFrame(
        data,
        columns=[
            "Column Name",
            "Number of Unique Values",
            "Unique Values",
            "Number of Null Values",
            "Proportion of Null Values",
            "dtype"
        ],
    ).sort_values(by="Number of Unique Values")




#~~~~~~~~~~~~~~~
# SPLIT
#~~~~~~~~~~~~~~~
def split_data(df):
    '''
    split continuouse data into train, validate, test; No target variable

    argument: df

    return: train, validate, test
    '''

    train_val, test = train_test_split(df,
                                   train_size=0.8,
                                   random_state=1108,
                                   )
    train, validate = train_test_split(train_val,
                                   train_size=0.7,
                                   random_state=1108,
                                   )
    
    print(f'Train: {len(train)/len(df)}')
    print(f'Validate: {len(validate)/len(df)}')
    print(f'Test: {len(test)/len(df)}')
    

    return train, validate, test


#~~~~~~~~~~~~~~~
# Visuals
#~~~~~~~~~~~~~~~
def corr_heat(df, drops):
    '''Creates a heatmap off of the dataset
    
    arguments: df, 'drop items'
    
    returns: heatmap visualization'''
    sns.heatmap(df.drop(columns=drops).corr(), center=1)
    plt.title('Correlation Heatmap')

def catcont_four_graphs(df, var1, var2, hue):
    """ Plots 4 graphs for visual representation of 1 continuous and 1 categorical variable
    
    arguments: df, var1, var2, hue, example input --> df, 'bedrooms', 'home_value', 'county'
    
    returns: 1 bar chart, 2 histograms, 1 boxplot"""
    #barplot
    plt.figure(figsize=(20,10))
    plt.suptitle(f'{var1} & {var2}')
    plt.subplot(221)
    sns.barplot(data=df, x=df[var1], y=df[var2], hue=hue)
    plt.title(f'Number of {var1} to {var2}')
    plt.xlabel(f'Number of {var1}')
    plt.ylabel(f'{var2}')
    
    #histo plot
    plt.subplot(222)
    sns.histplot(df, x=df[var1], bins=20)
    plt.title(f'Count of {var1}')
    plt.xlabel(f'Number of {var1}')
    
    #histo plot
    plt.subplot(223)
    sns.histplot(df, x=df[var2], bins=20)
    plt.title(f'Count of {var2}')
    plt.xlabel(f'Number of {var2}')
    
    #boxplot
    plt.subplot(224)
    sns.boxplot(data=df, x=var1, y=var2, hue=hue)
    plt.grid(False)
    plt.xlabel(f'Number of {var1}')
    plt.ylabel(f'{var2}')
    plt.show()
    


def contcont_four_graphs(df, var1, var2, hue):
    """ Plots 4 graphs for visual representation of 2 continuous variables
    
    arguments: df, var1, var2, hue, example input --> df, 'bedrooms', 'home_value', 'county'
    
    returns: 1 scatterplot, 2 histograms, 1 boxplot"""
    plt.figure(figsize=(20,10))
    plt.suptitle(f'{var1} & {var2}')
    plt.subplot(221)
    sns.scatterplot(data=df, x=df[var1], y=df[var2], hue=hue)
    plt.title(f'Number of {var1} to {var2}')
    plt.xlabel(f'Number of {var1}')
    plt.ylabel(f'{var2}')
    
    # print('~~~~~~~~~~~~~~~~~~~~~')
    plt.subplot(222)
    sns.histplot(df, x=df[var1], bins=20)
    plt.title(f'Count of {var1}')
    plt.xlabel(f'Number of {var1}')
    
    # print('~~~~~~~~~~~~~~~~~~~~~')
    plt.subplot(223)
    sns.histplot(df, x=df[var2], bins=20)
    plt.title(f'Count of {var2}')
    plt.xlabel(f'Number of {var2}')
    
    # print('~~~~~~~~~~~~~~~~~~~~~')
    plt.subplot(224)
    sns.boxplot(data=df, x=var1, hue=hue)
    plt.grid(False)
    plt.xlabel(f'Number of {var1}')
    plt.ylabel(f'{var2}')
    plt.show()
    # print('~~~~~~~~~~~~~~~~~~~~~')

def act_poly3_hist(y_train, pred_pr):
    '''Returns a histogram for poly 3 model'''
    plt.hist(y_train, color='blue', alpha=.5, label="Actual")
    
    plt.hist(pred_pr, color='pink', alpha=.5, label="Polynomial 3Deg")

    plt.xlabel("Home Value")
    plt.ylabel("Number of Houses")
    plt.title("Comparing the Distribution of Actual to Predicted House Values")
    plt.legend()
    plt.show()

def act_ploy2_hist(y_train, pred_pr):
    ''' Returns a histogram for poly 2 model'''
    plt.hist(y_train, color='blue', alpha=.5, label="Actual")
    
    plt.hist(pred_pr, color='green', alpha=.5, label="Polynomial 2Deg")
    

    plt.xlabel("Home Value")
    plt.ylabel("Number of Houses")
    plt.title("Comparing the Distribution of Actual to Predicted House Values")
    plt.legend()
    plt.show()

def act_lr_lar_hist(y_train, pred_lr1, pred_lars):
    '''Creates a histogram for line reg and lasso lars
    
    returns: histogram'''
    plt.hist(y_train, color='blue', alpha=.5, label="Actual")
    plt.hist(pred_lr1, color='red', alpha=.5, label="LinearRegression")
    plt.hist(pred_lars, color='grey', alpha=.5, label="LassoLars")

    plt.xlabel("Home Value")
    plt.ylabel("Number of Houses")
    plt.title("Comparing the Distribution of Actual to Predicted House Values")
    plt.legend()
    plt.show()

def act_poly3_hist_test(y_test, pred_test_pr):
    '''Takes 2 vars and creates a histogram to visualize test and actual data
    
    returns: histogram'''
    plt.hist(y_test, color='blue', alpha=.5, label="Actual")
    plt.hist(pred_test_pr, color='green', alpha=.5, label="Polynomial 3Deg")

    plt.xlabel("Home Value")
    plt.ylabel("Number of Houses")
    plt.title("Comparing the Distribution of Actual to Predicted Test House Values")
    plt.legend()
    plt.show()








# Stat Functions


# create stats functions
def spear_stat(var1, var2):
    '''Takes 2 variables from a dataframe and determines if they are dependent on each other or not
    
    arguments: var1, var2
    
    returns: print statements'''
    stat, p = spearmanr(var1, var2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent')
    else:
        print('Probably dependent')


def ttest_stat(var1,var2):
    '''Takes 2 variables from a dataframe and determines if they have the same
    distribution or not.
    
    arguments: var1, var2
    
    returns: print statements'''
    # t-test
    

    stat, p = ttest_ind(var1, var2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')






#~~~~~~~~~~~~~~~
# Model 1
#~~~~~~~~~~~~~~~
def model_prep_zillow(train, validate, test, target):
    # Make Dummies
    dummy_list = ['county']
    dummy_df = pd.get_dummies(train[dummy_list], dtype=int, drop_first=True)
    train_prepd = pd.concat([train, dummy_df], axis=1)

    dummy_list = ['county']
    dummy_df = pd.get_dummies(validate[dummy_list], dtype=int, drop_first=True)
    validate_prepd = pd.concat([validate, dummy_df], axis=1)

    dummy_list = ['county']
    dummy_df = pd.get_dummies(test[dummy_list], dtype=int, drop_first=True)
    test_prepd = pd.concat([test, dummy_df], axis=1)

    X_train = train_prepd[['bedrooms', 'bathrooms', 'finished_area']]
    y_train = train_prepd[target]

    X_validate = validate[['bedrooms', 'bathrooms', 'finished_area']]
    y_validate = validate_prepd[target]

    X_test = test[['bedrooms', 'bathrooms', 'finished_area']]
    y_test = test_prepd[target]


    return X_train, y_train, X_validate, y_validate, X_test, y_test
    

def model_prep_zillow_scaled(X_train, X_validate, X_test):
    '''Takes the X train, validate, and test and fits them to a RobustScaler
    
    arguments: X_train, X_validate, X_test
    
    returns: X_train_scaled, X_validate_scaled, X_test_scaled'''
    # makes a copy of the dataframes
    X_train_scaled = X_train.copy()
    X_valid_scaled = X_validate.copy()
    X_test_scaled = X_test.copy()

    columns_to_scale = ['bedrooms', 'bathrooms', 'finished_area']

    scaler = RobustScaler()

    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_valid_scaled[columns_to_scale] = scaler.transform(X_validate[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    return X_train_scaled, X_valid_scaled, X_test_scaled

def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2


def get_best_feat_rfe(df,df2, y, n):
    '''Takes in  train and validate, the y_train, and number of features you want to see for best selection'''

    #intial ML model
    lr1 = LinearRegression()

    #make it
    rfe = RFE(lr1, n_features_to_select=n)

    #fit it
    rfe.fit(df, y)

    #use it on train
    X_train_rfe = rfe.transform(df)

    #use it on validate
    X_val_rfe = rfe.transform(df2)
    print('selected top feature:', rfe.get_feature_names_out())
    return X_train_rfe, X_val_rfe

def get_lr(df, df2, y, y2, n):
    '''linear regression model 1 takes train and validate, y_train, y-validate and 3 features'''
    #intial ML model
    lr1 = LinearRegression()

    #make it
    rfe = RFE(lr1, n_features_to_select=n)

    #fit it
    rfe.fit(df, y)

    #use it on train
    X_train_rfe = rfe.transform(df)

    #use it on validate
    X_val_rfe = rfe.transform(df2)

    print('selected top feature:', rfe.get_feature_names_out())

    lr1.fit(X_train_rfe, y)

    #use the thing (make predictions)
    pred_lr1 = lr1.predict(X_train_rfe)
    pred_val_lr1 = lr1.predict(X_val_rfe)

    #train
    metrics_reg(y, pred_lr1)

    #validate
    rmse, r2 = metrics_reg(y2, pred_val_lr1)


    return rmse, r2, pred_lr1

#~~~~~~~~~~~~~~~
# Model 2
#~~~~~~~~~~~~~~~
def get_lasso(df, df2, y, y2):
    '''lassolars model'''
    #make it
    lars = LassoLars(alpha=1)

    #fit it
    lars.fit(df, y)

    #use it
    pred_lars = lars.predict(df)
    pred_val_lars = lars.predict(df2)

    #train
    metrics_reg(y, pred_lars)

    #validate
    rmse, r2 = metrics_reg(y2, pred_val_lars)
    
    return rmse, r2, pred_lars


#~~~~~~~~~~~~~~~
# Model 3
#~~~~~~~~~~~~~~~
def get_poly(df,df2,y,y2,n):
    '''polynomial regression model'''
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=n)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(df)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(df2)
    

    #make it
    pr = LinearRegression()

    #fit it
    pr.fit(X_train_degree2, y)

    #use it
    pred_pr = pr.predict(X_train_degree2)
    pred_val_pr = pr.predict(X_validate_degree2)

    #train
    metrics_reg(y, pred_pr)
    #validate
    rmse, r2 = metrics_reg(y2, pred_val_pr)
    return rmse, r2, pred_pr



#~~~~~~~~~~~~~~~
# Test Best
#~~~~~~~~~~~~~~~
def get_poly_test(df, df2, df3, y, y2, y3,n):
    '''takes train, validate, and test sets, and the number of degrees for polynomial regression model'''
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=n)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(df)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(df2)
    X_test_degree = pf.transform(df3)

    #make it
    pr = LinearRegression()

    #fit it
    pr.fit(X_train_degree2, y)

    #use it
    pred_pr = pr.predict(X_train_degree2)
    pred_val_pr = pr.predict(X_validate_degree2)
    pred_test_pr = pr.predict(X_test_degree)
    
    #train
    metrics_reg(y, pred_pr)
    #validate
    metrics_reg(y2, pred_val_pr)
    #test
    rmse, r2 = metrics_reg(y3, pred_test_pr)
    return rmse, r2, pred_test_pr