
"""
export_model.py
Trains a RandomForest pipeline on /mnt/data/Cars.csv and saves app/model.joblib.
Run: python export_model.py
"""
import os, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

DATA = "/mnt/data/Cars.csv"
APP_DIR = "/mnt/data/app"
OUT = os.path.join(APP_DIR, "model.joblib")
assert os.path.exists(DATA), f"Dataset not found at {DATA}"

df = pd.read_csv(DATA)

# Cleaning rules (assignment)
df = df[~df['fuel'].isin(['CNG','LPG'])]
df = df[df['owner'] != 'Test Drive Car']

owner_map = {'First Owner':1,'Second Owner':2,'Third Owner':3,'Fourth & Above Owner':4}
df['owner_num'] = df['owner'].map(owner_map)

def _num(series, sufs):
    s = series.copy().astype(str)
    for suf in sufs: s = s.str.replace(suf,'',regex=False)
    return pd.to_numeric(s, errors='coerce')

df['mileage_num']   = _num(df['mileage'], [' kmpl',' km/kg'])
df['engine_num']    = _num(df['engine'],  [' CC'])
df['max_power_num'] = _num(df['max_power'], [' bhp'])

df['brand'] = df['name'].str.split().str[0]
df.loc[df['brand']=='Land','brand'] = 'Land Rover'
df.loc[df['brand']=='Ashok','brand'] = 'Ashok Leyland'

df = df.drop(columns=['name','torque','mileage','engine','max_power'])

features_num = ['year','km_driven','mileage_num','engine_num','max_power_num','seats','owner_num']
features_cat = ['fuel','seller_type','transmission','brand']

X = df[features_num + features_cat].copy()
y = np.log(df['selling_price'].astype(float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))])

pre = ColumnTransformer([('num', num, features_num), ('cat', cat, features_cat)])

rf = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=42)
pipe = Pipeline([('preprocess', pre), ('model', rf)]).fit(X_train, y_train)

os.makedirs(APP_DIR, exist_ok=True)
joblib.dump(pipe, OUT)
print(f"Saved pipeline to {OUT}")
