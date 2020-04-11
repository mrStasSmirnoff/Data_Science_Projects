import logging as log
import re


def time_features_encoding(dataframe):
    """
    
    This function searches for datatime columns of a shape %Y-%m-%d and split them onto three corresponding columns
    of type "integer". This would be most useful for tree-based algorithms. Additionally, in the case of boosting it is always
    beneficial for one-hot encode if possible, and for that aside of transformed dataframe itself the function returns
    a list of new columns which can be later pd.get_dummies in pandas.
    
    """

    time_features_list = list(dataframe.select_dtypes(include=["datetime64"]).columns)

    if len(time_features_list) == 0:
        log.info("Dataframe does not contain time features")

    for time_feature in time_features_list:
        dataframe[time_feature + '_' + 'Year'] = dataframe[time_feature].apply(
            lambda x: int(str(x)[0:4]) if x == x else np.nan)
        dataframe[time_feature + '_' + 'Month'] = dataframe[time_feature].apply(
            lambda x: int(str(x)[5:7]) if x == x else np.nan)
        dataframe[time_feature + '_' + 'Day'] = dataframe[time_feature].apply(
            lambda x: int(str(x)[8:10]) if x == x else np.nan)
        dataframe = dataframe.drop(time_feature, axis=1)

    new_time_feature_names = dataframe.filter(regex='Year|Month|Day').columns.tolist()

    return dataframe, new_time_feature_names
