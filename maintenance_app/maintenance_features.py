"""Additional as-of-time predictors. No future event, outcome, ID or total quota is used."""
import maintenance_ml as ml
import numpy as np
import pandas as pd

def add_features(frame,sets,events):
    if 'combined_history' in sets and all(c in frame for c in sets['combined_history']):return frame,sets
    frame=frame.copy();sets={k:list(v) for k,v in sets.items()};added=[]
    grouped=frame.groupby(ml.ID,sort=False)
    first=grouped[ml.TIME].transform('min')
    frame['Observed history days']=(frame[ml.TIME]-first).dt.total_seconds()/86400;added.append('Observed history days')
    frame['Operating exposure since first reading']=frame['Cumulative Operating Hours']-grouped['Cumulative Operating Hours'].transform('first');added.append('Operating exposure since first reading')
    health=[]
    for signal in ['Inspection Score (0-100)','Bearing Temperature (C)','Vibration RMS (mm/s)','Motor Load (%)']:
        prior_mean=grouped[signal].transform(lambda s:s.shift(1).rolling(12,min_periods=3).mean())
        prior_std=grouped[signal].transform(lambda s:s.shift(1).rolling(12,min_periods=3).std())
        c=signal+' / deviation from prior baseline';frame[c]=frame[signal]-prior_mean;health.append(c)
        c=signal+' / standardized deviation';frame[c]=((frame[signal]-prior_mean)/prior_std.replace(0,np.nan)).clip(-12,12);health.append(c)
    event_names=['Completed breakdowns 30d','Completed breakdowns 90d','Completed breakdowns 180d','Completed breakdowns since history began','Days since completed breakdown','Median past breakdown interval days','Breakdown recurrence pressure','Past breakdowns per observed month']
    values=np.full((len(frame),len(event_names)),np.nan)
    e=events.copy()
    for c in ['Maintenance Start Timestamp','Maintenance End Timestamp']:e[c]=pd.to_datetime(e[c])
    e=e[e['Maintenance Type']=='breakdown'];groups={eid:g.sort_values('Maintenance Start Timestamp') for eid,g in e.groupby(ml.ID)}
    for eid,g in frame.groupby(ml.ID,sort=False):
        indexes=g.index.to_numpy();es=groups.get(eid)
        values[indexes,:4]=0;values[indexes,7]=0
        if es is None or es.empty:continue
        times=g[ml.TIME].to_numpy(dtype='datetime64[ns]')[:,None]
        starts=es['Maintenance Start Timestamp'].to_numpy(dtype='datetime64[ns]')[None,:]
        ends=es['Maintenance End Timestamp'].to_numpy(dtype='datetime64[ns]')[None,:]
        known=ends<=times
        for j,d in enumerate([30,90,180]):values[indexes,j]=(known&(starts>times-np.timedelta64(d,'D'))).sum(axis=1)
        values[indexes,3]=(known&(starts>=first.loc[indexes].to_numpy(dtype='datetime64[ns]')[:,None])).sum(axis=1)
        elapsed=np.where(known,(times-ends)/np.timedelta64(1,'D'),np.inf).min(axis=1);elapsed[np.isinf(elapsed)]=np.nan
        values[indexes,4]=elapsed
        for i,index in enumerate(indexes):
            seen=starts[0,known[i]]
            if len(seen)>=3:
                median=float(np.median(np.diff(seen)/np.timedelta64(1,'D')));values[index,5]=median
                if median>0:values[index,6]=elapsed[i]/median
        values[indexes,7]=values[indexes,3]*30/np.maximum(30,frame.loc[indexes,'Observed history days'].to_numpy())
    for i,c in enumerate(event_names):frame[c]=values[:,i]
    core=sets['core'];sets['sensor_trends']=ml.CAT+['Inspection Score (0-100)','Bearing Temperature (C)','Vibration RMS (mm/s)','Motor Load (%)','Pressure (bar)','Operating Hours Last 24h','Equipment Age (days)']+health
    sets['event_history']=ml.CAT+['Hours Since Last Synthetic Maintenance','Prior Synthetic Breakdowns 30d']+added+event_names
    sets['combined_history']=core+added+health+event_names
    for names in sets.values():
        assert len(names)==len(set(names)) and not set(names).intersection(ml.FORBIDDEN)
    return frame,sets
