import pandas as pd
import numpy as np
import re
from html import unescape
from sklearn.preprocessing import OneHotEncoder
import math


def remove_duplicates(df):
    """
    Remove duplicate rows from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with duplicate rows removed.
    """
    df = df.drop_duplicates()
    return df


def unify_manufacturer(df, column_name="manufactor"):
    """
    Standardize the manufacturer names in the specified column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the column containing manufacturer names.
    
    Returns:
    pd.DataFrame: DataFrame with unified manufacturer names.
    """
    df[column_name] = df[column_name].replace("Lexsus", "לקסוס")
    return df


def modify_model_name(df, column_name="model"):
    """
    Clean and modify model names by removing manufacturer names and years.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the column containing model names.
    
    Returns:
    pd.DataFrame: DataFrame with cleaned model names.
    """
    manufacturers = ['יונדאי', 'ניסאן', 'סוזוקי', 'טויוטה', 'קיה', 'אאודי', 'סובארו',
       'מיצובישי', 'מרצדס', 'ב.מ.וו', 'אופל', 'הונדה', 'פולקסווגן',
       'שברולט', 'מאזדה', 'וולוו', 'סקודה', 'פורד', 'לקסוס', 'קרייזלר',
       'סיטרואן', "פיג'ו", 'רנו', 'דייהטסו', 'מיני', 'אלפא רומיאו']
    
    pattern = r'\b(?:' + '|'.join(manufacturers) + r')\b'
    
    df[column_name] = df[column_name].apply(lambda x: re.sub(pattern, '', x.strip().replace('`', "'")).strip())
    
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s*\(\d{4}\)', '', x))
    
    return df


def unify_model_names(df, column_name="model"):
    """
    Standardize specific model names based on predefined replacements.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the column containing model names.
    
    Returns:
    pd.DataFrame: DataFrame with standardized model names.
    """
    replacements = {
        "CIVIC": "סיוויק",
        "C-Class קופה": "C-Class",
        "C-CLASS קופה": "C-Class",
        "E- CLASS": "E-Class",
        "E-Class קופה / קבריולט": "E-Class",
        "מיטו / MITO": "מיטו",
        "גראנד, וויאג`ר": "גראנד, וויאג'ר",
        "וויאג`ר": "וויאג'ר",
        "GS300": "GS300",
        "IS250": "IS250",
        "IS300H": "IS300H",
        "IS300h": "IS300H",
        "RC": "RC",
        "CT200H": "CT200H",
        "ג'וק JUKE": "ג'וק",
        "סיטיגו / Citygo": "סיטיגו",
        "ACCORD": "אקורד"}
    
    df.loc[:, column_name] = df[column_name].replace(replacements)    
    
    return df


def fill_Gear(df, column_name="Gear", model_column="model", manufactor_column="manufactor"):
    """
    Fill missing values in the Gear column based on the most common values for the model and manufacturer.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the column containing Gear information.
    model_column (str): Name of the column containing model names.
    manufactor_column (str): Name of the column containing manufacturer names.
    
    Returns:
    pd.DataFrame: DataFrame with missing Gear values filled.
    """
    df[column_name] = df[column_name].replace("אוטומט", "אוטומטית")
    
    most_common_model = df.groupby(model_column)[column_name].agg(lambda x: x.mode().iloc[0]
                                                                                 if not x.mode().empty
                                                                                 else None)
    
    df[column_name] = df.apply(lambda row: most_common_model[row[model_column]]
                               if pd.isnull(row[column_name])
                               else row[column_name], axis=1)
    
    most_common_manufactor = df.groupby(manufactor_column)[column_name].agg(lambda x: x.mode().iloc[0]
                                                                                           if not x.mode().empty
                                                                                           else None)
    
    df[column_name] = df.apply(lambda row: most_common_manufactor[row[manufactor_column]]
                               if pd.isnull(row[column_name])
                               else row[column_name], axis=1)
    
    return df


def fill_Color(df, column_name="Color"):
    """
    Fill missing or 'None' values in the Color column with 'לא מוגדר'.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the column containing Color information.
    
    Returns:
    pd.DataFrame: DataFrame with missing Color values filled.
    """
    replacements = {
        "None": "לא מוגדר",
        np.nan: "לא מוגדר"}
    
    df[column_name] = df[column_name].replace(replacements)
    
    return df


def fill_Km(df, column_name="Km"):
    """
    Convert Km values to numeric and fill missing values based on the average kilometers driven per year.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the column containing Km information.
    
    Returns:
    pd.DataFrame: DataFrame with filled Km values.
    """
    try:
        df[column_name] = pd.to_numeric(df[column_name].str.replace(',', ''), errors="coerce")
    except AttributeError:
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    
    df[column_name] = df[column_name].replace(0, np.nan)
    
    if df[column_name].isnull().any():
        df[column_name] = df[column_name].fillna(((df["Km"] / (2024 - df["Year"])).mean()) * (2024 - df["Year"]))
    
    return df


def modify_Km(df, column_name="Km"):
    """
    Ensure Km values are in thousands if they are less than 1000.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the column containing Km information.
    
    Returns:
    pd.DataFrame: DataFrame with modified Km values.
    """
    df[column_name] = df[column_name].apply(lambda x: x * 1000 if x < 1000 else x)
    
    return df


def data_filling(df):
    """
    Perform various data cleaning and filling operations on the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with filled and cleaned data.
    """
    df = remove_duplicates(df)
    df = unify_manufacturer(df)
    df = modify_model_name(df)
    df = unify_model_names(df)
    df = fill_Gear(df)
    df = fill_Color(df)
    df = fill_Km(df)
    df = modify_Km(df)    
    df = df.reset_index(drop=True)
    
    return df


def vehicle_usage(df):
    """
    Calculate vehicle age, average kilometers per year, and categorize usage.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with vehicle age, average kilometers per year, and usage category.
    """
    df["Year"] = df["Year"].astype(int)
    
    current_year = pd.Timestamp.now().year
    
    df["Vehicle_Age"] = current_year - df["Year"]
    
    df["Avg_Km_Per_Year"] = (df["Km"] / df["Vehicle_Age"]).round().astype(int)
    
    conditions = [
        (df["Avg_Km_Per_Year"] > 15000),
        (df["Avg_Km_Per_Year"] < 10000),
        (df["Avg_Km_Per_Year"].between(10000, 15000, inclusive="left"))]
    
    choices = ["Above Range", "Below Range", "In Range"]
    
    df["Usage_Category"] = np.select(conditions, choices, default="In Range")
    
    return df


def vehicle_age(df):
    """
    Determine the age category of vehicles based on their age.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with age category.
    """
    current_year = pd.Timestamp.now().year
    
    df['Vehicle_Age'] = current_year - df['Year']
    
    def age_category(age):
        if age < 3:
            return 'Very New'
        elif 3 <= age <= 8:
            return 'New'
        elif 8 < age < 15:
            return 'Reasonable'
        elif 15 < age < 19:
            return 'Old'
        elif 19 <= age <= 30:
            return 'Very Old'
        else:
            return 'Collectors Vehicle'
    
    df['Age_Category'] = df['Vehicle_Age'].apply(age_category)
    
    return df


def add_features(df):
    """
    Add additional features to the DataFrame including vehicle usage and age categories.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with added features.
    """
    df = vehicle_usage(df)
    df = vehicle_age(df)
    
    return df


def group_low_frequency_categories(df, min_frequency=30):
    """
    Group categories with low frequency into an 'Other' category.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    min_frequency (int): Minimum frequency threshold for categorization.
    
    Returns:
    pd.DataFrame: DataFrame with low-frequency categories grouped.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_cols:
        value_counts = df[col].value_counts()
        to_replace = value_counts[value_counts < min_frequency].index
        df[col] = df[col].apply(lambda x: 'Other' if x in to_replace else x)
    
    return df


def encode_categorical_variables(df):

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    encoder = OneHotEncoder(drop='first')

    encoded_categorical = encoder.fit_transform(df[categorical_cols])

    feature_names = encoder.get_feature_names_out(categorical_cols)

    encoded_df = pd.DataFrame(encoded_categorical.toarray(), columns=feature_names)

    df_encoded = df.drop(categorical_cols, axis=1)

    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    return df_encoded


def delete_columns(df):
    """
    Delete specified columns from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with specified columns removed.
    """
    columns_to_delete = ["capacity_Engine", "Engine_type", "Curr_ownership", "Area", "City", "Pic_num", "Cre_date", "Repub_date", "Description", "Test", "Supply_score"]
    
    existing_columns_to_delete = [col for col in columns_to_delete if col in df.columns]
    
    if existing_columns_to_delete:
        df = df.drop(existing_columns_to_delete, axis=1)
    
    return df


def prepare_data(df):
    """
    Prepare data by performing cleaning, feature engineering, and encoding.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: Processed DataFrame ready for modeling.
    """
    df = data_filling(df)
    df = delete_columns(df)
    df = group_low_frequency_categories(df)
    df = encode_categorical_variables(df)
    
    return df
