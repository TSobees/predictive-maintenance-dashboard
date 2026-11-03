"""Approved service schedules and current inspection prompts, independent of synthetic cycles."""
from datetime import datetime
from zoneinfo import ZoneInfo
import math,uuid,sqlite3
import pandas as pd
import forecasting as fc
import maintenance_ml as ml

ACTIVE_POLICY='catch_more'
STRATEGY={'key':ACTIVE_POLICY,'label':'Catch more failures','definition':'Prioritize finding failure risk. Uses the frozen validation threshold targeting at least 80% recall; more inspections are expected. Scheduled work is flagged independently of model risk.'}

def local_now():return pd.Timestamp(datetime.now(ZoneInfo('Africa/Lagos')).replace(tzinfo=None))
def date_value(value,name='Date'):
    try:t=pd.Timestamp(value)
    except (TypeError,ValueError):raise ValueError(name+' must be a valid local date and time.')
    if pd.isna(t) or t.tzinfo is not None:raise ValueError(name+' must be a local date and time without a timezone suffix.')
    return t

def number(value,name,integer=False):
    if value is None or value=='':return None
    try:n=float(value)
    except (TypeError,ValueError):raise ValueError(name+' must be a number.')
    if not math.isfinite(n) or n<0 or (integer and (n<1 or n%1 or n>3650)):raise ValueError(name+' is outside the allowed range.')
    return int(n) if integer else n

class MaintenancePlanning:
    def init_planning(self):
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS maintenance_plans(
              id TEXT PRIMARY KEY,equipment_id TEXT NOT NULL,maintenance_type TEXT NOT NULL,task TEXT NOT NULL,
              due_at TEXT,due_meter REAL,interval_days INTEGER,interval_hours REAL,enabled INTEGER NOT NULL,
              revision INTEGER NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS maintenance_completions(
              id TEXT PRIMARY KEY,plan_id TEXT NOT NULL,event_id TEXT NOT NULL,completed_at TEXT NOT NULL,
              completed_meter REAL,next_due_at TEXT,next_due_meter REAL,created_at TEXT NOT NULL);''')

    def plans(self):
        with self.connect() as db:
            db.row_factory=sqlite3.Row
            rows=[dict(r) for r in db.execute('SELECT * FROM maintenance_plans ORDER BY enabled DESC,due_at,id')]
        for r in rows:
            r['enabled']=bool(r['enabled'])
            r['equipment']=self.assets_by_id.loc[r['equipment_id'],'Equipments'];r['station']=self.assets_by_id.loc[r['equipment_id'],'Stations']
        return rows

    def save_plan(self,payload):
        from services import TYPES
        eid=payload.get('equipment_id');kind=payload.get('maintenance_type');task=str(payload.get('task','')).strip()
        if eid not in self.assets_by_id.index or kind not in TYPES:raise ValueError('Select registered equipment and a valid work type.')
        if len(task)<5 or len(task)>300:raise ValueError('Describe the planned work in 5–300 characters.')
        due=str(date_value(payload['due_at'],'Due date')) if payload.get('due_at') else None
        meter=number(payload.get('due_meter'),'Due operating-hours meter');days=number(payload.get('interval_days'),'Repeat days',True);hours=number(payload.get('interval_hours'),'Repeat operating hours')
        if hours is not None and hours<=0:raise ValueError('Repeat operating hours must be greater than zero.')
        if due is None and meter is None:raise ValueError('Supply a due date or an operating-hours meter limit.')
        if days is not None and due is None:raise ValueError('A repeating calendar plan needs its first due date.')
        if hours is not None and meter is None:raise ValueError('A repeating hours plan needs its first due meter reading.')
        enabled=payload.get('enabled',True)
        if not isinstance(enabled,bool):raise ValueError('Enabled must be true or false.')
        with self.lock,self.connect() as db:
            existing=db.execute('SELECT revision,created_at FROM maintenance_plans WHERE id=?',(payload.get('id'),)).fetchone() if payload.get('id') else None
            if payload.get('id') and existing is None:raise ValueError('This plan was not found.')
            if existing and payload.get('revision')!=existing[0]:raise ValueError('This plan changed. Refresh before editing it.')
            if existing:
                identity=db.execute('SELECT equipment_id,maintenance_type FROM maintenance_plans WHERE id=?',(payload['id'],)).fetchone()
                has_history=db.execute('SELECT 1 FROM maintenance_completions WHERE plan_id=? LIMIT 1',(payload['id'],)).fetchone()
                if has_history and (eid,kind)!=tuple(identity):raise ValueError('Create a new plan to change equipment or work type after a completion.')
            now=self.now();row={'id':payload.get('id') or str(uuid.uuid4()),'equipment_id':eid,'maintenance_type':kind,'task':task,'due_at':due,'due_meter':meter,'interval_days':days,'interval_hours':hours,'enabled':int(enabled),'revision':existing[0]+1 if existing else 1,'created_at':existing[1] if existing else now,'updated_at':now}
            if existing:
                db.execute('UPDATE maintenance_plans SET '+','.join(k+'=?' for k in row if k!='id')+' WHERE id=?',[row[k] for k in row if k!='id']+[row['id']])
            else:db.execute('INSERT INTO maintenance_plans ('+','.join(row)+') VALUES('+','.join('?' for _ in row)+')',list(row.values()))
        row['enabled']=bool(row['enabled'])
        return row

    def complete_plan(self,plan_id,payload):
        with self.lock,self.connect() as db:
            db.row_factory=sqlite3.Row;p=db.execute('SELECT * FROM maintenance_plans WHERE id=?',(plan_id,)).fetchone()
            if p is None or not p['enabled']:raise ValueError('This plan is no longer active.')
            if payload.get('revision')!=p['revision']:raise ValueError('This plan changed or was already completed. Refresh the schedule.')
            done=date_value(payload.get('completed_at'),'Completion time')
            if done>local_now():raise ValueError('Completion time cannot be in the future.')
            previous=db.execute('SELECT completed_at,completed_meter FROM maintenance_completions WHERE plan_id=? ORDER BY completed_at DESC LIMIT 1',(plan_id,)).fetchone()
            if previous and done<=pd.Timestamp(previous['completed_at']):raise ValueError('Completion time must follow the previous completion.')
            meter=number(payload.get('completed_meter'),'Completion operating-hours meter')
            if p['due_meter'] is not None and meter is None:raise ValueError('Record the actual operating-hours meter at completion.')
            if previous and meter is not None and previous['completed_meter'] is not None and meter<previous['completed_meter']:raise ValueError('The completion meter must not decrease.')
            downtime=number(payload.get('downtime_hours',0),'Downtime hours') or 0
            event_start=done-pd.DateOffset(hours=downtime)
            event=self.add_event({'equipment_id':p['equipment_id'],'maintenance_type':p['maintenance_type'],'occurred_at':str(event_start),'operation_stopped':payload.get('operation_stopped',False),'downtime_hours':payload.get('downtime_hours',0),'notes':payload.get('notes','')},db=db)
            next_date=str(done+pd.DateOffset(days=p['interval_days'])) if p['interval_days'] else None
            next_meter=meter+p['interval_hours'] if p['interval_hours'] else None
            enabled=int(next_date is not None or next_meter is not None)
            db.execute('UPDATE maintenance_plans SET due_at=?,due_meter=?,enabled=?,revision=revision+1,updated_at=? WHERE id=?',(next_date,next_meter,enabled,self.now(),plan_id))
            result={'id':str(uuid.uuid4()),'plan_id':plan_id,'event_id':event['id'],'completed_at':str(done),'completed_meter':meter,'next_due_at':next_date,'next_due_meter':next_meter,'created_at':self.now()}
            db.execute('INSERT INTO maintenance_completions VALUES(?,?,?,?,?,?,?,?)',list(result.values()))
        return result

    def maintenance_status(self,equipment_id=None,now=None):
        now=local_now() if now is None else date_value(now)
        if equipment_id and equipment_id not in self.assets_by_id.index:raise ValueError('Unknown equipment ID.')
        all_plans=self.plans();plans=[p for p in all_plans if not equipment_id or p['equipment_id']==equipment_id]
        context=self.context();context=context if not equipment_id else context[context[ml.ID]==equipment_id]
        features,_=fc.build_features(context,self.assets,self.forecast_events());latest=features.groupby(ml.ID,sort=False).tail(1).copy()
        predictions=self.forecasts.predict_frame(latest,policy=ACTIVE_POLICY,explain=False)
        events=self.events();event_times={}
        for e in events:event_times[e['equipment_id']]=max(event_times.get(e['equipment_id'],pd.Timestamp.min),pd.Timestamp(e['occurred_at'])+pd.DateOffset(hours=e['downtime_hours']))
        # Only user-supplied measurements may drive operating-hours due limits.
        actual={}
        for r in self.observations():
            if r[ml.ID] not in actual or pd.Timestamp(r[ml.TIME])>pd.Timestamp(actual[r[ml.ID]][ml.TIME]):actual[r[ml.ID]]=r
        with self.connect() as db:
            for eid,t,meter in db.execute('SELECT p.equipment_id,c.completed_at,c.completed_meter FROM maintenance_completions c JOIN maintenance_plans p ON p.id=c.plan_id WHERE c.completed_meter IS NOT NULL'):
                if eid not in actual or pd.Timestamp(t)>pd.Timestamp(actual[eid][ml.TIME]):actual[eid]={ml.TIME:t,'Cumulative Operating Hours':meter}
        by_asset={};readings=[]
        for (_,r),pred in zip(latest.iterrows(),predictions):
            eid=r[ml.ID];as_of=pd.Timestamp(r[ml.TIME]);age=max(0,(now-as_of).total_seconds()/86400)
            synthetic=str(r.get('Record Source','')).lower().startswith('synthetic')
            work_after=event_times.get(eid,pd.Timestamp.min)>as_of
            fresh=age<=7 and not work_after and not synthetic;active=r['Operating Status'] in ['Running','Degraded']
            flags=[v['days'] for v in pred['risks'] if v['inspect']] if fresh and active else []
            interval=min(flags) if flags else None
            item={'equipment_id':eid,'equipment':self.assets_by_id.loc[eid,'Equipments'],'station':self.assets_by_id.loc[eid,'Stations'],'reading_at':str(as_of),'reading_age_days':round(age,1),'operating_status':r['Operating Status'],'reading_required':not fresh,'model_alert':bool(flags),'risk_horizon_days':interval,'risk_review_by':str(as_of+pd.DateOffset(days=interval)) if interval else None,'seven_day_probability':pred['risks'][0]['probability'] if fresh and active else None,'note':'Add current measured readings before using the model for scheduling.' if not fresh else 'Inspect within the earliest flagged forecast window.' if flags else 'Follow approved maintenance tasks; no current model inspection flag.','plans_count':sum(p['enabled'] for p in plans if p['equipment_id']==eid)}
            by_asset[eid]=item;readings.append(item)
        for p in plans:
            real=actual.get(p['equipment_id']);meter=float(real['Cumulative Operating Hours']) if real else None
            meter_fresh=bool(real and (now-pd.Timestamp(real[ml.TIME])).total_seconds()<=7*86400)
            until=(pd.Timestamp(p['due_at'])-now).total_seconds()/86400 if p['due_at'] else None
            remaining=p['due_meter']-meter if p['due_meter'] is not None and meter is not None else None
            overdue=bool((until is not None and until<=0) or (remaining is not None and remaining<=0))
            soon=bool((until is not None and 0<until<=7) or (meter_fresh and remaining is not None and 0<remaining<=24))
            p.update({'status':'Inactive' if not p['enabled'] else 'Overdue' if overdue else 'Due soon' if soon else 'Reading needed' if p['due_meter'] is not None and not meter_fresh else 'Scheduled','days_until_due':round(until,2) if until is not None else None,'hours_remaining':remaining,'actual_meter':meter,'meter_reading_at':real[ml.TIME] if real else None,'meter_refresh_required':p['due_meter'] is not None and not meter_fresh})
        plans.sort(key=lambda p:({'Overdue':0,'Due soon':1,'Reading needed':2,'Scheduled':3,'Inactive':4}[p['status']],p['due_at'] or '9999',p['equipment_id']))
        readings.sort(key=lambda r:(0 if r['model_alert'] else 1 if r['reading_required'] else 2,r['risk_horizon_days'] or 999,r['equipment_id']))
        summary={'overdue_tasks':sum(p['status']=='Overdue' for p in plans),'due_soon_tasks':sum(p['status']=='Due soon' for p in plans),'model_inspection_flags':sum(r['model_alert'] for r in readings),'readings_needed':sum(r['reading_required'] for r in readings),'equipment_without_plans':sum(r['plans_count']==0 for r in readings),'active_plans':sum(p['enabled'] for p in plans)}
        return {'as_of':str(now),'strategy':STRATEGY,'summary':summary,'plans':plans,'equipment':readings,'alert_delivery':'In-app flags refresh every minute while the app is open. Reopen the app to recalculate overdue work.','rules':'Overdue: approved date or measured meter limit reached. Due soon: within 7 calendar days or 24 measured operating hours. Recurrence starts from recorded completion. No meter values are extrapolated from synthetic history.'}
