"""Causal event features, coherent multi-horizon risks and review policies."""
from pathlib import Path
import json
import maintenance_ml as ml
import gradient_boosting as gb
from maintenance_features import add_features
import numpy as np
import pandas as pd

HORIZONS=[7,14,30]
POLICIES={
 'balanced':{'label':'Balanced','definition':'Maximize balanced accuracy on validation: equal importance to failure recall and non-failure specificity.'},
 'catch_more':{'label':'Catch more failures','definition':'Highest validation threshold achieving at least 80% recall; more inspections are expected.'},
 'fewer_inspections':{'label':'Fewer inspections','definition':'Validation alert budget of at most 10%; actual future workload may differ.'}}

def build_features(history,assets,events):
    frame,sets=ml.build_features(history,assets)
    events=events.copy();events['Maintenance Start Timestamp']=pd.to_datetime(events['Maintenance Start Timestamp'])
    groups={id:g.sort_values('Maintenance Start Timestamp') for id,g in events.groupby(ml.ID)}
    added=['Prior completed work 30d','Prior completed work 90d','Prior breakdowns 90d','Hours since previous breakdown','Hours since previous lubrication','Hours since previous inspection']
    values=np.full((len(frame),len(added)),np.nan)
    for id,rows in frame.groupby(ml.ID,sort=False):
        group=groups.get(id)
        indexes=rows.index.to_numpy()
        if group is None or group.empty:
            values[indexes,:3]=0;continue
        times=rows[ml.TIME].to_numpy(dtype='datetime64[ns]')[:,None]
        starts=group['Maintenance Start Timestamp'].to_numpy(dtype='datetime64[ns]')[None,:]
        ends=pd.to_datetime(group['Maintenance End Timestamp']).to_numpy(dtype='datetime64[ns]')[None,:]
        kinds=group['Maintenance Type'].to_numpy()[None,:]
        known=ends<=times
        values[indexes,0]=(known&(starts>times-np.timedelta64(30,'D'))).sum(axis=1)
        values[indexes,1]=(known&(starts>times-np.timedelta64(90,'D'))).sum(axis=1)
        values[indexes,2]=(known&(starts>times-np.timedelta64(90,'D'))&(kinds=='breakdown')).sum(axis=1)
        for j,kind in enumerate(['breakdown','lubrication','inspection'],3):
            hours=np.where(known&(kinds==kind),(times-starts)/np.timedelta64(1,'h'),np.inf).min(axis=1)
            hours[np.isinf(hours)]=np.nan;values[indexes,j]=hours
    for j,name in enumerate(added):frame[name]=values[:,j]
    sets['enriched']=sets['lags']+added
    return add_features(frame,sets,events)

def coherent_probabilities(matrix):
    """Freeze this projection before evaluation: risk cannot decrease with horizon."""
    values=np.asarray(matrix,dtype=float)
    if values.ndim!=2 or values.shape[1]!=3:raise ValueError('Expected 7/14/30-day probability columns.')
    return np.maximum.accumulate(np.clip(values,0,1),axis=1)

class ForecastEngine:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'forecast_manifest.json').read_text())
        self.events=pd.read_csv(self.directory/'maintenance_events.csv')
        self.events['Maintenance Start Timestamp']=pd.to_datetime(self.events['Maintenance Start Timestamp'])
        self.events['Maintenance End Timestamp']=pd.to_datetime(self.events['Maintenance End Timestamp'])
        self.is_gradient=self.manifest.get('algorithm')==gb.ALGORITHM
        if not self.is_gradient:raise ValueError('This public demo supports GradientBoostingClassifier only.')
        self.backend=gb
        self.explanation_method=gb.EXPLANATION if self.is_gradient else 'SHAP values on the controlling CatBoost model raw score.'
        self.models={h:self.backend.load_models(self.directory,self.manifest['horizons'][str(h)]) for h in HORIZONS}

    def predict_frame(self,frame,policy=None,explain=True,explain_horizon=7):
        policy=policy or self.manifest.get('default_policy','balanced')
        if policy not in self.manifest.get('policy_definitions',POLICIES):raise ValueError('Choose a valid alert policy.')
        ps=[]
        for h in HORIZONS:
            spec=self.manifest['horizons'][str(h)]
            _,p=self.backend.predict_models(self.models[h],frame[spec['features']],spec['calibration']);ps.append(p)
        raw=np.column_stack(ps);projected=coherent_probabilities(raw)
        drivers=[[] for _ in range(len(frame))]
        selected_index=HORIZONS.index(explain_horizon)
        providers=np.argmax(raw[:,:selected_index+1],axis=1)
        if explain and len(frame):
            for provider in np.unique(providers):
                indexes=np.flatnonzero(providers==provider);h=HORIZONS[provider];spec=self.manifest['horizons'][str(h)]
                _,ds=self.backend.driver_explanations(self.models[h],frame.iloc[indexes][spec['features']])
                for index,ds_row in zip(indexes,ds):drivers[index]=ds_row
        results=[]
        for i,(_,row) in enumerate(frame.iterrows()):
            risks=[]
            for j,h in enumerate(HORIZONS):
                spec=self.manifest['horizons'][str(h)];threshold=spec['policies'][policy]['threshold']
                risks.append({'days':h,'probability':float(projected[i,j]),'threshold':threshold,'inspect':bool(projected[i,j]>=threshold),
                    'horizon_ends':str(row[ml.TIME]+pd.DateOffset(days=h)),
                    'coherence_adjusted':bool(abs(projected[i,j]-raw[i,j])>1e-12),
                    'test_metrics':spec['policies'][policy]['test_metrics']})
            results.append({'risks':risks,'drivers':drivers[i],'explanation_model_horizon_days':HORIZONS[providers[i]],'explanation_method':self.explanation_method})
        return results

def cycle_summary(events,equipment_id,as_of,source='Synthetic simulation events'):
    """Describe past recurrence; do not turn a sparse history into an RUL claim."""
    work=events.copy();work['Maintenance Start Timestamp']=pd.to_datetime(work['Maintenance Start Timestamp'])
    work=work[(work[ml.ID]==equipment_id)&(work['Maintenance Start Timestamp']<=as_of)]
    failures=work[work['Maintenance Type']=='breakdown'].sort_values('Maintenance Start Timestamp')
    intervals=failures['Maintenance Start Timestamp'].diff().dt.total_seconds().dropna()/86400
    adequate=len(intervals)>=2
    return {'source':source,'breakdown_count':len(failures),'interval_count':len(intervals),
        'median_interval_days':float(intervals.median()) if adequate else None,
        'interval_range_days':[float(intervals.min()),float(intervals.max())] if adequate else None,
        'last_breakdown':str(failures['Maintenance Start Timestamp'].max()) if len(failures) else None,
        'status':'Descriptive only; too few events for a validated maintenance cycle' if adequate else 'Insufficient breakdown intervals to estimate a cycle',
        'predicted_failure_date':None,
        'note':'Past median spacing is not remaining useful life or an OEM service interval. The machine can fail before or after this spacing.'}
