from sklearn import preprocessing 
import pandas as pd
import logging as log


def to_label_encode_cat_data(dataframe):
    """
    Label encoding
    """
    label_encoder = preprocessing.LabelEncoder() 
    cat_features = get_list_of_cat_features(dataframe)
    
    for column in cat_features:  
        dataframe[column] = label_encoder.fit_transform(dataframe[column])
     
    return dataframe


def to_one_hot_encode_cat_data(dataframe):
    """
    One-Hot encoding
    """
    cat_features = get_list_of_cat_features(dataframe)

    if len(cat_features) > 0:
        df_ohe = pd.get_dummies(dataframe, columns = cat_features)
    else:
        log.info("Dataframe returned without encoding")
        df_ohe = dataframe

    return df_ohe


def get_list_of_cat_features(dataframe):
    """
    Get list of categorical features
    """

    cat_features_list = list(dataframe.select_dtypes(include=["object"]).columns)
    
    if len(cat_features_list) == 0:
        log.info("There is no categorical features in the dataframe")
       
    return cat_features_list