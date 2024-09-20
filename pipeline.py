import pandas as pd
import numpy as np
from datetime import datetime
import re
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

from sklearn.metrics import root_mean_squared_error

def load_data(filepath):
    data = pd.read_csv(filepath)
    data.replace('–', np.nan, inplace=True)
    data.replace('not supported', np.nan, inplace=True)
    return data

def remove_outliers(data):
    data.drop(index=data[(data['model_year']==1974)|(data['price'] > 1500000)].index, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def drop_columns(data, column):
    data.drop(columns=column, inplace=True)
    return data

def fill_fuel_type(data):
    electric_brands = ['Tesla', 'Lucid', 'Rivian']
    data.loc[data['brand'].isin(electric_brands), 'fuel_type'] = 'Electric'
        
    fuel_type_conditions = [
        (data['engine'].str.contains('Electric|Battery')) & (data['fuel_type'].isna()),
        (data['engine'].str.contains('Hybrid')) & (data['fuel_type'].isna()),
        (data['engine'].str.contains('Gasoline')) & (data['fuel_type'].isna()),
        (data['engine'].str.contains('Diesel')) & (data['fuel_type'].isna()),
        (data['engine'].str.contains('Hydrogen')) & (data['fuel_type'].isna()),
        (data['model'].str.contains('EV|Electric')) & (data['fuel_type'].isna())
        ]
    fuel_type_values = ['Electric', 'Hybrid', 'Gasoline', 'Diesel', 'Hydrogen', 'Electric']
        
    for condition, value in zip(fuel_type_conditions, fuel_type_values):
        data.loc[condition, 'fuel_type'] = value
        
    data.loc[data['fuel_type'].isna(), 'fuel_type'] = 'Gasoline'

    return data

def correct_brand_names(data):
    data.loc[data['brand'] == 'Land', 'brand'] = 'Land Rover'
    return data

def fill_missing_values(data):
    columns_to_fill = ['accident', 'clean_title', 'ext_col', 'int_col']
    for column in columns_to_fill:
        data[column].fillna('Unknown', inplace=True)
    return data

def format_strings(data, column):
    data[column] = data[column].str.title()
    return data

def create_dictionary(data, group, column):
    mode = data.groupby(group)[column].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index()
    return mode.set_index(group)[column].to_dict()

def fill_from_dictionary(data, column, dictionary, type=None):

    if type == 'brand':
        data['brand'] = data['model'].map(dictionary)
    else:
        data[column] = data.apply(lambda row: dictionary[row['model']] if pd.isna(row[column]) else row[column], axis=1)
    
    return data

def save_dictionary(dictionary, filename):
    with open(filename, "wb") as file:
        pickle.dump(dictionary, file)
    print("Pickle Saved!")

def add_age_and_bins(data):
    data['age'] = datetime.now().year - data['model_year']
    data['age_bins'] = pd.cut(data['age'], bins=[-np.inf, 2, 5, 10, 15, 20, np.inf], labels=['Fairly New', '3-5 Years', '6-10 Years', '11-15 Years', '16-20 Years', 'Old'])
    return data

def extract_value(data):

    def extract_horsepower(hp):
        try:
            pattern = r'\d+\.\d+HP'
            match = re.search(pattern, hp)
            hp_string = match.group()
            horsepower = hp_string.replace('.0HP', '')
            horsepower = int(horsepower)
            return horsepower
        except:
            return np.nan
    
    def extract_engine_liter(liter):
        try:
            pattern = r'\d+\.\d+L'
            match = re.search(pattern, liter)
            liter_string = match.group()
            liter = liter_string.replace('L', '')
            liter = float(liter)
            return liter
        except:
            return np.nan
    
    def extract_cylinder(cyn):
        try:
            pattern = r'\d+ Cylinder'
            match = re.search(pattern, cyn)
            cyn_string = match.group()
            cylinder = cyn_string.replace(' Cylinder', '')
            cylinder = int(cylinder)
            return cylinder
        except:
            return np.nan
    
    def extract_gear(gear):
        try:
            pattern = r'^\d+'
            match = re.search(pattern, gear)
            return int(match.group())
        except:
            return np.nan
    
    data['hp'] = data['engine'].apply(extract_horsepower)
    data['engine_liter'] = data['engine'].apply(extract_engine_liter)
    data['cylinder'] = data['engine'].apply(extract_cylinder)
    data['num_gear'] = data['transmission'].apply(extract_gear)

    return data

def clean_new_features(train, data, dictionary):
    hp_median = train['hp'].median()
    engine_liter_median = train['engine_liter'].median()
    cylinder_median = train['cylinder'].median()

    def fill_missing_values(row):
        row['hp'] = row['hp'] if not pd.isna(row['hp']) else hp_median
        row['engine_liter'] = row['engine_liter'] if not pd.isna(row['engine_liter']) else engine_liter_median
        row['cylinder'] = row['cylinder'] if not pd.isna(row['cylinder']) else cylinder_median
        return row

    filled_df = data.apply(fill_missing_values, axis=1)

    filled_df['hp'] = filled_df.apply(lambda row: dictionary[row['model']] if row['model'] in dictionary and row['hp'] != dictionary[row['model']] else row['hp'], axis=1)

    filled_df.loc[((filled_df['transmission'] == 'Single-Speed Fixed Gear') & (filled_df['num_gear'].isna())), 'num_gear'] = 1

    filled_df['hp'].fillna(hp_median, inplace=True)

    return filled_df

def clean_transmission(train, data):
    def remove_speed(trans):
        pattern = r'\d+-[Ss][Pp][Ee][Ee][Dd]\s*'
        return re.sub(pattern, '', trans).strip()

    def replace_transmission(trans):
        if 'A/T' in trans:
            return trans.replace('A/T', 'Automatic')
        elif 'M/T' in trans:
            return trans.replace('M/T', 'Manual')
        elif 'Variable' in trans:
            return trans.replace('Variable', 'CVT')
        else:
            return trans

    data['transmission'] = data['transmission'].apply(remove_speed)
    data['transmission'] = data['transmission'].apply(replace_transmission)

    correct_transmission = {
        'CVT Transmission': 'CVT',
        'Automatic CVT': 'CVT',
        'Electronically Controlled Automatic with O': 'Automatic with Overdrive',
        'F': np.nan,
        'CVT-F': 'CVT',
        '2': np.nan,
        '6 Speed At/Mt': 'Automated Manual',
        '': np.nan,
        'AT': 'Automatic',
        'SCHEDULED FOR OR IN PRODUCTION': np.nan,
        '6 Speed Mt': 'Manual'
        }

    transmission_mode = train['transmission'].mode()[0]
    gear_mode = train['num_gear'].mode()[0]

    data['transmission'] = data['transmission'].replace(correct_transmission)
    data['transmission'].fillna(transmission_mode, inplace=True)
    data['num_gear'].fillna(gear_mode, inplace=True)
    return data

def target_variable(data, column):
    target = data[column]
    log_target = np.log(target)
    return log_target

def drop_target(data):
    new_data = data.drop(columns=['price'])
    return new_data

def scale_data(data):
    pre_scale_data = data.select_dtypes(include=['float', 'int'])
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(pre_scale_data), columns=pre_scale_data.columns)
    pickle.dump(scaler, open("scaler.pickle", "wb"))
    return scaled_data

def encode_data(data):
    data['accident'] = data['accident'].map(lambda x: 1 if x == 'At least 1 accident or damage reported' else 0)
    data['clean_title'] = data['clean_title'].map(lambda x: 1 if x == 'Yes' else 0)
    pre_enc_data = data.select_dtypes(include=['object'])
    encoded_data = pd.get_dummies(pre_enc_data, dtype='int')
    return encoded_data

def combine_data(scaled_data, encoded_data):
    data = pd.concat([encoded_data, scaled_data], axis=1)
    pickle.dump(data.columns, open("selected_features.pickle", "wb"))
    return data
    
def model_training(train, target):
    X = train
    y = target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cbr = CatBoostRegressor()
    cbr.fit(X_train, y_train)
    cbr_pred = cbr.predict(X_test)

    rmse = root_mean_squared_error(np.exp(y_test), np.exp(cbr_pred))
    pickle.dump(cbr, open("catboost.pickle", "wb"))

    print(f"The RMSE of the model is {rmse}.")