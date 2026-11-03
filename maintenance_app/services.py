"""Model-backed scoring and local record collection for the maintenance app."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'data'/'model'
sys.path.insert(0,str(MODEL))
import maintenance_ml as ml
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import forecasting as fc
from planning import MaintenancePlanning,ACTIVE_POLICY,STRATEGY

TYPES={
 'condition-based':'Issue present; operations can continue.',
 'breakdown':'Complete stop of operation caused by a fault.',
 'lubrication':'Lubricant servicing of designated points.',
 'inspection':'Inspect equipment for irregular operation.',
 'support':'Ancillary work such as welding or fabrication.',
 'refurbishment':'Restore a completely idle or out-of-work machine.',
 'modification':'Upgrade a machine that is still working.'}
FIELDS=[
 {'key':'Operating Hours Last 24h','label':'Operating hours · last 24h','unit':'hours','min':0,'max':24,'required':True},
 {'key':'Cumulative Operating Hours','label':'Cumulative operating hours','unit':'hours','min':0,'required':True},
 {'key':'Hours Since Last Synthetic Maintenance','label':'Hours since last maintenance','unit':'hours','min':0,'required':True},
 {'key':'Prior Synthetic Breakdowns 30d','label':'Unplanned breakdowns · past 30 days','unit':'count','min':0,'step':1,'required':True},
 {'key':'Inspection Score (0-100)','label':'Inspection score (higher = worse condition)','unit':'0–100','min':0,'max':100,'required':True},
 {'key':'Equipment Age (days)','label':'Equipment age','unit':'days','min':0,'step':1,'required':True},
 {'key':'Bearing Temperature (C)','label':'Bearing temperature','unit':'°C','required':False},
 {'key':'Vibration RMS (mm/s)','label':'Vibration RMS','unit':'mm/s','min':0,'required':False},
 {'key':'Motor Load (%)','label':'Motor load','unit':'%','min':0,'max':150,'required':False},
 {'key':'Pressure (bar)','label':'Pressure','unit':'bar','min':0,'required':False}]

def records(frame):
    values=frame.astype(object).where(pd.notna(frame),None).to_dict(orient='records')
    return json.loads(json.dumps(values,default=lambda x:x.isoformat() if isinstance(x,pd.Timestamp) else ml.json_default(x),allow_nan=False))

def timestamp(value):
    if not isinstance(value,str) or not value.strip():raise ValueError('Supply an observation timestamp.')
    t=pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is not None:raise ValueError('Use local dates and times without a timezone suffix (Africa/Lagos).')
    return t

class MaintenanceService(MaintenancePlanning):
    def __init__(self,db_path=None):
        self.manifest=json.loads((MODEL/'model_manifest.json').read_text())
        self.evaluation=json.loads((MODEL/'evaluation.json').read_text())
        self.assets=pd.read_csv(MODEL/'asset_metadata.csv')
        self.assets_by_id=self.assets.set_index(ml.ID)
        self.history=pd.read_csv(MODEL/'historical_context.csv');self.history[ml.TIME]=pd.to_datetime(self.history[ml.TIME])
        self.latest=pd.read_csv(MODEL/'latest_snapshot_predictions.csv')
        forecast_dir=ROOT/'data'/'forecast_models'
        self.forecasts=fc.ForecastEngine(forecast_dir) if (forecast_dir/'forecast_manifest.json').exists() else None
        if self.forecasts is None:raise ValueError('Run tools/train_demo.py to build the public demonstration models.')
        self.models=[]
        if self.forecasts:
            display=self.forecasts.manifest.get('model_display_name','CatBoost')
            spec=self.forecasts.manifest['horizons']['7'];new=spec['policies'][ACTIVE_POLICY]['test_metrics']
            self.manifest={**self.manifest,'alert_threshold':spec['policies'][ACTIVE_POLICY]['threshold'],
                'threshold_policy':STRATEGY['definition'],
                'source_sha256':self.forecasts.manifest['source_sha256'],'source_workbook':self.forecasts.manifest.get('source_workbook','training_dataset.xlsx'),'dataset_revision':self.forecasts.manifest.get('dataset_revision','original simulation'),
                'model_type':f"{len(spec['model_files'])}-model {display} forecast",'trees_per_model':spec['trees_per_model'],'forecast_version':self.forecasts.manifest.get('version',2),'algorithm':self.forecasts.manifest.get('algorithm','CatBoostClassifier'),'explanation_method':self.forecasts.explanation_method,
                'features':spec['features'],'calibration':spec['calibration'],'selected_parameters':spec['configuration'],
                'trained_until':spec['periods']['training']['end'],'model_files':['../forecast_models/'+f for f in spec['model_files']]}
            self.evaluation={**self.evaluation,'test':new,'transfer_test':spec['transfer_metrics'][ACTIVE_POLICY],
                'source_sha256':self.forecasts.manifest['source_sha256'],'source_workbook':self.forecasts.manifest.get('source_workbook','training_dataset.xlsx'),'dataset_revision':self.forecasts.manifest.get('dataset_revision'),
                'dataset_summary':self.forecasts.manifest.get('dataset_summary',{}),
                'comparison':{display+' · '+STRATEGY['label']:new},
                'search_trials':self.forecasts.manifest.get('search_trials',24),'cross_validation_folds':3,
                'periods':{('threshold_validation' if k=='validation' else k):v for k,v in spec['periods'].items()}}
            for legacy in ['delivery_checks','runtime_seconds','calibration_candidates','ensemble_choice']:self.evaluation.pop(legacy,None)
            self.evaluation.pop('bootstrap_95_percent',None)
            self.evaluation.pop('selected_trial',None)
            self.evaluation['validation_policy']=spec['policies'][ACTIVE_POLICY]['validation_metrics']
            self.evaluation['calibration_choice']=spec['calibration']['method']
            self.latest=pd.read_csv(forecast_dir/'latest_snapshot_balanced.csv',float_precision='round_trip')
            self.latest['Alert threshold']=self.manifest['alert_threshold']
            active=self.latest['Operating Status'].isin(['Running','Degraded'])
            self.latest['Alert decision']=np.where(~active,'Not scored: equipment inactive',np.where(self.latest['Seven-day failure probability']>=self.manifest['alert_threshold'],'Inspect: above alert threshold','Routine monitoring: below threshold'))
        self.lock=threading.RLock()
        self.db_path=Path(db_path or ROOT/'data'/'maintenance_records.sqlite3')
        self.db_path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS observations(
                id TEXT PRIMARY KEY,equipment_id TEXT NOT NULL,observed_at TEXT NOT NULL,payload TEXT NOT NULL,
                prediction TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(equipment_id,observed_at));
                CREATE TABLE IF NOT EXISTS maintenance_events(id TEXT PRIMARY KEY,equipment_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL,maintenance_type TEXT NOT NULL,operation_stopped INTEGER NOT NULL,
                downtime_hours REAL NOT NULL,notes TEXT NOT NULL,created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS inspection_queue(equipment_id TEXT PRIMARY KEY,status TEXT NOT NULL,notes TEXT NOT NULL,updated_at TEXT NOT NULL);''')

        self.init_planning()

    @contextmanager
    def connect(self):
        db=sqlite3.connect(str(self.db_path),timeout=20)
        try:
            with db:yield db
        finally:db.close()
    def now(self):return datetime.now(ZoneInfo('Africa/Lagos')).isoformat(timespec='seconds')

    def overview(self):
        with self.connect() as db:
            observations=db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
            events=db.execute('SELECT COUNT(*) FROM maintenance_events').fetchone()[0]
            queue={r[0]:{'status':r[1],'notes':r[2]} for r in db.execute('SELECT equipment_id,status,notes FROM inspection_queue')}
        latest=records(self.latest.sort_values('Seven-day failure probability',ascending=False,na_position='last',kind='stable'))
        for r in latest:r['Queue status']=queue.get(r[ml.ID],{}).get('status','Not reviewed')
        stations=[]
        for station,g in self.latest.groupby('Station'):
            stations.append({'station':station,'assets':len(g),'alerts':int((g['Seven-day failure probability']>=self.manifest['alert_threshold']).sum()),'mean_probability':float(g['Seven-day failure probability'].mean())})
        return {'equipment':latest,'stations':stations,'evaluation':self.evaluation,
                'manifest':self.manifest,'counts':{'assets':len(latest),'alerts':sum(r['Seven-day failure probability'] is not None and r['Seven-day failure probability']>=self.manifest['alert_threshold'] for r in latest),'observations':observations,'events':events},
                'fields':FIELDS,'maintenance_types':TYPES,'forecast_options':{'strategy':STRATEGY,'policies':{ACTIVE_POLICY:STRATEGY},'horizons':fc.HORIZONS,'ready':self.forecasts is not None},
                'forecast_evaluation':self.public_forecast_manifest() if self.forecasts else None,'optimization_review':json.loads((self.forecasts.directory/'optimization_review.json').read_text()) if self.forecasts and (self.forecasts.directory/'optimization_review.json').exists() else None}

    def asset(self,equipment_id):
        if equipment_id not in self.assets_by_id.index:raise ValueError('Unknown equipment ID.')
        meta=self.assets_by_id.loc[equipment_id].to_dict()
        selected=self.latest[self.latest[ml.ID]==equipment_id]
        h=self.history[self.history[ml.ID]==equipment_id].sort_values(ml.TIME)
        with self.connect() as db:
            saved=db.execute('SELECT payload FROM observations WHERE equipment_id=? ORDER BY observed_at',(equipment_id,)).fetchall()
            queue=db.execute('SELECT status,notes,updated_at FROM inspection_queue WHERE equipment_id=?',(equipment_id,)).fetchone()
        return {'metadata':meta,'snapshot':records(selected)[0],'history':records(h),
                'observations':[json.loads(r[0]) for r in saved],
                'queue':{'status':queue[0],'notes':queue[1],'updated_at':queue[2]} if queue else {'status':'Not reviewed','notes':''}}

    def context(self):
        with self.connect() as db:saved=[json.loads(r[0]) for r in db.execute('SELECT payload FROM observations')]
        if not saved:return self.history.copy()
        extra=pd.DataFrame(saved);extra[ml.TIME]=pd.to_datetime(extra[ml.TIME])
        for field in FIELDS:extra[field['key']]=pd.to_numeric(extra[field['key']],errors='raise')
        return pd.concat([self.history,extra],ignore_index=True).drop_duplicates([ml.ID,ml.TIME],keep='last')

    def score(self,incoming,mode='new',save=False,policy=ACTIVE_POLICY,horizon=7):
        if policy!=ACTIVE_POLICY or horizon not in fc.HORIZONS:raise ValueError('Select a valid alert policy and forecast horizon.')
        if mode not in ['new','historical']:raise ValueError('Choose new observations or historical replay.')
        if mode=='historical' and save:raise ValueError('Historical replay cannot be saved as new observed data.')
        incoming=incoming.copy()
        if incoming.empty or len(incoming)>5000:raise ValueError('Supply between 1 and 5,000 observations.')
        for key in [ml.ID,ml.TIME,'Operating Status']+[f['key'] for f in FIELDS if f['required']]:
            if key not in incoming:raise ValueError('Missing input column: '+key)
        incoming[ml.TIME]=incoming[ml.TIME].map(lambda x:timestamp(str(x)))
        incoming[ml.ID]=incoming[ml.ID].astype(str).str.strip()
        if incoming[ml.ID].isin(self.assets_by_id.index).all()==False:raise ValueError('An equipment ID is not in the register.')
        if incoming.duplicated([ml.ID,ml.TIME]).any():raise ValueError('Duplicate equipment/timestamp rows.')
        if not incoming['Operating Status'].isin(['Running','Degraded','Idle','Under maintenance']).all():raise ValueError('Status must be Running, Degraded, Idle or Under maintenance.')
        for f in FIELDS:
            key=f['key']
            if key not in incoming:incoming[key]=np.nan
            incoming[key]=pd.to_numeric(incoming[key].replace('',np.nan),errors='raise')
            values=incoming[key]
            if f['required'] and values.isna().any():raise ValueError(f"{f['label']} is required.")
            if np.isinf(values.to_numpy(dtype=float)).any():raise ValueError(f"{f['label']} must be finite.")
            if 'min' in f and (values.dropna()<f['min']).any():raise ValueError(f"{f['label']} must be at least {f['min']}.")
            if 'max' in f and (values.dropna()>f['max']).any():raise ValueError(f"{f['label']} must not exceed {f['max']}.")
            if f.get('step')==1 and (values.dropna()%1!=0).any():raise ValueError(f"{f['label']} must be a whole number.")
        expected_station=incoming[ml.ID].map(self.assets_by_id['Stations'])
        if 'Station' in incoming and ((incoming['Station'].notna()) & incoming['Station'].ne(expected_station)).any():raise ValueError('Station does not match the equipment register.')
        incoming['Station']=expected_station
        incoming['Equipment']=incoming[ml.ID].map(self.assets_by_id['Equipments'])
        incoming['Record Source']='Historical replay' if mode=='historical' else 'User supplied; unverified observation'
        # Uploaded outcomes are discarded and can never become prediction inputs.
        allowed=[ml.ID,'Equipment','Station',ml.TIME,'Operating Status','Record Source']+[f['key'] for f in FIELDS]
        incoming=incoming[allowed]
        with self.lock:
            context=self.history.copy() if mode=='historical' else self.context()
            if mode=='new':
                last=context.groupby(ml.ID)[ml.TIME].max()
                if (incoming[ml.TIME]<=incoming[ml.ID].map(last)).any():raise ValueError('New observations must follow the latest saved reading for each asset. Select historical replay for old data.')
                if (incoming[ml.TIME]>pd.Timestamp(datetime.now(ZoneInfo('Africa/Lagos')).replace(tzinfo=None))).any():raise ValueError('An observation timestamp cannot be in the future.')
            combined=pd.concat([context,incoming],ignore_index=True).drop_duplicates([ml.ID,ml.TIME],keep='last')
            frame,_=fc.build_features(combined,self.assets,self.forecasts.events if mode=='historical' else self.forecast_events()) if self.forecasts else ml.build_features(combined,self.assets)
            keys=pd.MultiIndex.from_frame(incoming[[ml.ID,ml.TIME]])
            selected=frame[pd.MultiIndex.from_frame(frame[[ml.ID,ml.TIME]]).isin(keys)].copy()
            active=selected['Operating Status'].isin(['Running','Degraded'])
            result=selected[[ml.ID,'Equipment','Station',ml.TIME,'Operating Status']].copy()
            result['Forecast horizon ends']=selected[ml.TIME].map(lambda t:t+pd.DateOffset(days=horizon))
            result['Seven-day failure probability']=np.nan
            result['Selected horizon failure probability']=np.nan
            result['Alert threshold']=self.forecasts.manifest['horizons'][str(horizon)]['policies'][policy]['threshold'] if self.forecasts else self.manifest['alert_threshold']
            result['Alert decision']='Not scored: equipment inactive'
            result['Prediction context']='Historical simulation replay' if mode=='historical' else 'Supplied observations / demonstration model'
            explanations={}
            if active.any():
                if self.forecasts:
                    predictions=self.forecasts.predict_frame(selected.loc[active],policy=policy,explain_horizon=horizon)
                    p=np.array([r['risks'][0]['probability'] for r in predictions]);ds=[r['drivers'] for r in predictions]
                else:
                    X=selected.loc[active,self.manifest['features']]
                    _,p=ml.predict_models(self.models,X,self.manifest['calibration'])
                    _,ds=ml.driver_explanations(self.models,X)
                result.loc[active,'Seven-day failure probability']=p
                result.loc[active,'Selected horizon failure probability']=p
                result.loc[active,'Alert decision']=np.where(p>=self.manifest['alert_threshold'],'Inspect: above alert threshold','Routine monitoring: below threshold')
                for index,drivers in zip(selected.index[active],ds):explanations[index]=drivers
                if self.forecasts:
                    for index,prediction in zip(selected.index[active],predictions):
                        risk=next(r for r in prediction['risks'] if r['days']==horizon)
                        result.loc[index,'Selected horizon failure probability']=risk['probability']
                        result.loc[index,'Alert threshold']=risk['threshold']
                        result.loc[index,'Alert decision']='Inspect: above alert threshold' if risk['inspect'] else 'Routine monitoring: below threshold'
                        result.loc[index,'Forecast horizon ends']=pd.Timestamp(risk['horizon_ends'])
            result['Selected horizon days']=horizon;result['Alert policy']=fc.POLICIES[policy]['label']
            result['Model algorithm']=self.manifest.get('algorithm','CatBoostClassifier')
            result['Model source SHA-256']=self.manifest['source_sha256']
            output=records(result)
            for r,index in zip(output,result.index):r['drivers']=explanations.get(index,[])
            if self.forecasts:
                scored={i:p for i,p in zip(selected.index[active],predictions)} if active.any() else {}
                for r,index in zip(output,result.index):
                    r['horizon_risks']=scored.get(index,{}).get('risks',[])
                    r['explanation_model_horizon_days']=scored.get(index,{}).get('explanation_model_horizon_days')
                    r['explanation_method']=self.forecasts.explanation_method
            if save:
                payloads={ (r[ml.ID],r[ml.TIME]):r for r in records(incoming)}
                with self.connect() as db:
                    for r in output:
                        key=(r[ml.ID],r[ml.TIME]);payload=payloads[key]
                        db.execute('INSERT INTO observations VALUES(?,?,?,?,?,?)',(str(uuid.uuid4()),r[ml.ID],str(timestamp(r[ml.TIME])),json.dumps(payload),json.dumps(r),self.now()))
            return output

    def add_event(self,payload,db=None):
        equipment_id=payload.get('equipment_id');kind=payload.get('maintenance_type')
        if equipment_id not in self.assets_by_id.index:raise ValueError('Select a registered equipment item.')
        if kind not in TYPES:raise ValueError('Select a valid maintenance type.')
        t=timestamp(payload.get('occurred_at'))
        if t>pd.Timestamp(datetime.now(ZoneInfo('Africa/Lagos')).replace(tzinfo=None)):raise ValueError('The maintenance event cannot be in the future.')
        stopped=payload.get('operation_stopped')
        if not isinstance(stopped,bool):raise ValueError('Specify whether operations stopped.')
        if kind=='breakdown' and not stopped:raise ValueError('Breakdown means a complete stop of operation.')
        if kind=='condition-based' and stopped:raise ValueError('Condition-based means operations can continue with the issue.')
        downtime=float(payload.get('downtime_hours',0))
        if not np.isfinite(downtime) or downtime<0:raise ValueError('Downtime must be a nonnegative number.')
        if not stopped and downtime!=0:raise ValueError('Downtime must be zero when operations did not stop.')
        notes=str(payload.get('notes','')).strip()
        if len(notes)<5 or len(notes)>2000:raise ValueError('Describe the work or fault in 5–2,000 characters.')
        event={'id':str(uuid.uuid4()),'equipment_id':equipment_id,'occurred_at':str(t),'maintenance_type':kind,'operation_stopped':int(stopped),'downtime_hours':downtime,'notes':notes,'created_at':self.now()}
        if db is not None:db.execute('INSERT INTO maintenance_events VALUES(?,?,?,?,?,?,?,?)',tuple(event.values()))
        else:
            with self.connect() as connection:connection.execute('INSERT INTO maintenance_events VALUES(?,?,?,?,?,?,?,?)',tuple(event.values()))
        return event

    def events(self):
        with self.connect() as db:
            db.row_factory=sqlite3.Row
            rows=[dict(r) for r in db.execute('SELECT * FROM maintenance_events ORDER BY occurred_at DESC')]
        for r in rows:r['equipment']=self.assets_by_id.loc[r['equipment_id'],'Equipments'];r['station']=self.assets_by_id.loc[r['equipment_id'],'Stations']
        return rows

    def queue(self,payload):
        eid=payload.get('equipment_id');status=payload.get('status');notes=str(payload.get('notes','')).strip()
        if eid not in self.assets_by_id.index or status not in ['Not reviewed','Inspection planned','Reviewed']:raise ValueError('Select a valid asset and review status.')
        if len(notes)>2000:raise ValueError('Review notes must be at most 2,000 characters.')
        with self.connect() as db:db.execute('INSERT INTO inspection_queue VALUES(?,?,?,?) ON CONFLICT(equipment_id) DO UPDATE SET status=excluded.status,notes=excluded.notes,updated_at=excluded.updated_at',(eid,status,notes,self.now()))
        return {'equipment_id':eid,'status':status,'notes':notes}

    def public_forecast_manifest(self):
        import copy
        manifest=copy.deepcopy(self.forecasts.manifest)
        manifest['active_policy']=ACTIVE_POLICY;manifest['policy_definitions']={ACTIVE_POLICY:STRATEGY}
        manifest.pop('comparison_7d',None)
        for spec in manifest['horizons'].values():
            spec['policies']={ACTIVE_POLICY:spec['policies'][ACTIVE_POLICY]}
            spec['transfer_metrics']={ACTIVE_POLICY:spec['transfer_metrics'][ACTIVE_POLICY]}
        return manifest

    def observations(self):
        with self.connect() as db:return [json.loads(r[0]) for r in db.execute('SELECT payload FROM observations ORDER BY observed_at DESC')]

    def forecast_events(self):
        if not self.forecasts:return pd.DataFrame()
        rows=self.events();events=self.forecasts.events.copy()
        if rows:
            extra=pd.DataFrame([{ml.ID:r['equipment_id'],'Maintenance Type':r['maintenance_type'],
                'Maintenance Start Timestamp':pd.Timestamp(r['occurred_at']),
                'Maintenance End Timestamp':pd.Timestamp(r['occurred_at'])+pd.DateOffset(hours=r['downtime_hours']),
                'Operational Downtime (hrs)':r['downtime_hours']} for r in rows])
            events=pd.concat([events,extra],ignore_index=True)
        return events

    def equipment_forecast(self,equipment_id,policy=ACTIVE_POLICY,horizon=7):
        if not self.forecasts:raise ValueError('The multi-horizon forecast models are not available yet.')
        if equipment_id not in self.assets_by_id.index:raise ValueError('Unknown equipment ID.')
        if policy!=ACTIVE_POLICY or horizon not in fc.HORIZONS:raise ValueError('Select a valid policy and horizon.')
        context=self.context();history=context[context[ml.ID]==equipment_id].sort_values(ml.TIME)
        latest=history.iloc[-1];as_of=pd.Timestamp(latest[ml.TIME]);now=pd.Timestamp(datetime.now(ZoneInfo('Africa/Lagos')).replace(tzinfo=None))
        age=max(0,(now-as_of).total_seconds()/86400);stale=age>7
        observed=self.events()
        newer_work=any(e['equipment_id']==equipment_id and as_of<pd.Timestamp(e['occurred_at'])+pd.DateOffset(hours=e['downtime_hours'])<=now for e in observed)
        synthetic=str(latest.get('Record Source','')).lower().startswith('synthetic')
        refresh_required=stale or newer_work or synthetic
        source=str(latest.get('Record Source','Unknown'));active=latest['Operating Status'] in ['Running','Degraded']
        result={'equipment_id':equipment_id,'equipment':self.assets_by_id.loc[equipment_id,'Equipments'],'station':self.assets_by_id.loc[equipment_id,'Stations'],
            'as_of':str(as_of),'reading_source':source,'reading_age_days':round(age,1),'stale':stale,'freshness_limit_days':7,
            'status':latest['Operating Status'],'policy':policy,'policy_label':fc.POLICIES[policy]['label'],'selected_horizon_days':horizon,
            'synthetic_model':True,'current_forecast_available':active and not refresh_required,'refresh_required':refresh_required,'work_after_reading':newer_work,'risks':[],'drivers':[],
            'freshness_note':'Latest readings are older than the weekly training cadence. Results are historical projections; add current measured readings for a current forecast.' if stale else 'Forecast is anchored to the displayed observation, not to an invented new measurement.'}
        if synthetic:result['freshness_note']='This observation is simulated. Enter actual measured readings before using model prompts for current work planning.'
        if newer_work:result['freshness_note']='Maintenance was recorded after this observation. Update readings to reflect the equipment condition after that work.'
        if active:
            frame,_=fc.build_features(history,self.assets,self.forecast_events());row=frame.tail(1)
            result.update(self.forecasts.predict_frame(row,policy=policy,explain_horizon=horizon)[0])
        real=pd.DataFrame([{ml.ID:e['equipment_id'],'Maintenance Type':e['maintenance_type'],'Maintenance Start Timestamp':e['occurred_at']} for e in observed],columns=[ml.ID,'Maintenance Type','Maintenance Start Timestamp'])
        result['observed_cycle']=fc.cycle_summary(real,equipment_id,as_of,'User-entered maintenance records')
        result['simulation_cycle']=fc.cycle_summary(self.forecasts.events,equipment_id,as_of)
        flagged=[r['days'] for r in result['risks'] if r['inspect']]
        interval=min(flagged) if flagged else None
        result['review_plan']={'interval_days':interval if interval and active and not refresh_required else None,
            'review_by':str(as_of+pd.DateOffset(days=interval)) if interval and active and not refresh_required else None,
            'provisional_historical_interval_days':interval if active else None,
            'reason':'Update readings before scheduling from this model.' if refresh_required else 'Not scored while inactive.' if not active else 'Inspect within the earliest flagged forecast window.' if flagged else 'Follow approved schedules; no model inspection flag. No automatic 30-day service is assumed.',
            'method':'Planning rule based on model flags. This is not a learned optimal service interval, an OEM schedule, or an exact failure date.'}
        return result
