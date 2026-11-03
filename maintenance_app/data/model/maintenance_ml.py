"""Causal feature preparation for the public synthetic demonstration."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd

def demo_source_digest(contents):
    """Stable fingerprint of the three generated CSV inputs, in fixed order."""
    if isinstance(contents,(str,Path)):
        contents={name:(Path(contents)/name).read_bytes() for name in ['assets.csv','events.csv','observations.csv']}
    digest=hashlib.sha256()
    for name in ['assets.csv','events.csv','observations.csv']:
        content=contents[name]
        digest.update(name.encode('utf-8')+b'\0'+str(len(content)).encode('ascii')+b'\0'+content)
    return digest.hexdigest()
from scipy.special import expit

ID='Equipment ID'
TIME='Observation Timestamp'
TARGET='Unplanned Failure Next 7d'
CAT=['Station','Equipment Class','Operating Status']
NUM=['Operating Hours Last 24h','Cumulative Operating Hours','Hours Since Last Synthetic Maintenance',
     'Prior Synthetic Breakdowns 30d','Bearing Temperature (C)','Vibration RMS (mm/s)',
     'Motor Load (%)','Pressure (bar)','Inspection Score (0-100)','Equipment Age (days)',
     'Simulated Criticality (1-5)','Simulated Typical Daily Hours']
SIGNALS=['Inspection Score (0-100)','Bearing Temperature (C)','Vibration RMS (mm/s)',
         'Motor Load (%)','Pressure (bar)','Operating Hours Last 24h']
FORBIDDEN=['Next Failure Lead Time (hrs)','Failure Event ID Next 7d',TARGET,ID,'Equipment',
           'Record Source','Label Basis','Maintenance Type','Fault Description','Action Taken']

def build_features(history,assets):
    """Only current or past observations; no interpolation or future outcomes."""
    frame=history.copy()
    for col in [ID,TIME,'Station','Operating Status']+NUM[:10]:
        if col not in frame: raise ValueError(f'Missing required column: {col}')
    frame[TIME]=pd.to_datetime(frame[TIME],errors='raise')
    if frame[TIME].isna().any():raise ValueError('Observation timestamps must be populated')
    if frame.duplicated([ID,TIME]).any():raise ValueError('Duplicate equipment/timestamp observations')
    if assets[ID].duplicated().any():raise ValueError('Duplicate equipment IDs in register')
    registered_station=frame[ID].map(assets.set_index(ID)['Stations'])
    if (registered_station.notna() & frame['Station'].ne(registered_station)).any():
        raise ValueError('Observation station disagrees with the equipment register')
    meta=assets[[ID,'Equipment Class','Simulated Criticality (1-5)','Simulated Typical Daily Hours']]
    for c in meta.columns[1:]:
        if c in frame:frame=frame.drop(columns=c)
    frame=frame.merge(meta,on=ID,how='left',validate='many_to_one',indicator=True)
    if (frame['_merge']!='both').any():raise ValueError('Equipment ID not present in the asset register')
    frame=frame.drop(columns='_merge').sort_values([ID,TIME]).reset_index(drop=True)
    for col in NUM:frame[col]=pd.to_numeric(frame[col],errors='raise')
    if np.isinf(frame[NUM].to_numpy(dtype=float)).any():raise ValueError('Numeric readings must be finite or missing')
    for col in CAT:frame[col]=frame[col].fillna('Missing').astype(str)
    hours=frame['Operating Hours Last 24h']
    if hours.isna().any() or (~hours.between(0,24)).any():raise ValueError('Daily operating hours must be between 0 and 24')
    for c in ['Cumulative Operating Hours','Equipment Age (days)','Hours Since Last Synthetic Maintenance','Prior Synthetic Breakdowns 30d']:
        if frame[c].isna().any() or (frame[c]<0).any():raise ValueError(f'{c} must be populated and nonnegative')
    if not frame['Inspection Score (0-100)'].dropna().between(0,100).all():raise ValueError('Inspection score must be 0-100')
    lag_names=[]
    for signal in SIGNALS:
        group=frame.groupby(ID,sort=False)[signal]
        for suffix,fn in [
            (' / previous observation',lambda s:s.shift(1)),
            (' / change since previous',lambda s:s.diff()),
            (' / trailing 4 observations mean',lambda s:s.rolling(4,min_periods=1).mean()),
            (' / trailing 4 observations std',lambda s:s.rolling(4,min_periods=2).std())]:
            col=signal+suffix;frame[col]=group.transform(fn);lag_names.append(col)
    frame['Days since previous observation']=frame.groupby(ID)[TIME].diff().dt.total_seconds()/86400
    lag_names.append('Days since previous observation')
    core=CAT+NUM
    sets={'core':core,'lags':core+lag_names}
    for names in sets.values():
        assert not set(names).intersection(FORBIDDEN)
        assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in names if c not in CAT)
    return frame,sets

def calibration_predict(raw,calibration):
    raw=np.asarray(raw,dtype=float)
    if calibration['method']=='identity':return expit(raw)
    if calibration['method']=='sigmoid':return expit(calibration['slope']*raw+calibration['intercept'])
    if calibration['method']=='isotonic':return np.interp(raw,calibration['x'],calibration['y'])
    raise ValueError('Unknown calibration method')

def check_causal_features(history,assets,frame,sets):
    dates=np.sort(pd.to_datetime(history[TIME]).unique())
    cutoff=dates[len(dates)//2]
    prefix=history[pd.to_datetime(history[TIME])<=cutoff]
    short,_=build_features(prefix,assets)
    reference=frame[frame[TIME]<=cutoff].sort_values([ID,TIME]).reset_index(drop=True)
    pd.testing.assert_frame_equal(short[sets['lags']],reference[sets['lags']])
    poisoned=history.copy()
    for col in [TARGET,'Next Failure Lead Time (hrs)','Failure Event ID Next 7d']:
        if col in poisoned:poisoned[col]='POISONED FUTURE VALUE'
    other,_=build_features(poisoned,assets)
    pd.testing.assert_frame_equal(frame[sets['lags']],other[sets['lags']])

def json_default(value):
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,)):return float(value)
    if isinstance(value,(pd.Timestamp,np.datetime64)):return str(value)
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)

def save_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,default=json_default,allow_nan=False),encoding='utf-8')
