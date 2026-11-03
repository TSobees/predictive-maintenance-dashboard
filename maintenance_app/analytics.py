"""Read-only workbook analytics with validated atomic refresh and stable equipment IDs."""
from pathlib import Path
from datetime import datetime,date,time,timedelta
from zoneinfo import ZoneInfo
from collections import Counter,defaultdict
import hashlib,io,json,math,re,threading,zipfile,copy,csv
from maintenance_ml import demo_source_digest
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.datetime import from_excel

APP=Path(__file__).resolve().parent
LOCAL=ZoneInfo('Africa/Lagos')
TYPES=['condition-based','breakdown','lubrication','inspection','support','refurbishment','modification']
def now():return datetime.now(LOCAL).isoformat(timespec='seconds')
def text(v):return '' if v is None else str(v).strip()
def key(v):return re.sub(r'[^a-z0-9]','',text(v).lower())
def numeric(v):
    if isinstance(v,bool):return None
    try:n=float(v)
    except (ValueError,TypeError):return None
    return n if math.isfinite(n) and n>=0 else None

def stamp(v):
    if v is None or v=='':return None
    try:
        if isinstance(v,(int,float)) and not isinstance(v,bool):v=from_excel(v)
        if isinstance(v,datetime):d=v
        elif isinstance(v,date):d=datetime.combine(v,time())
        elif isinstance(v,str):
            try:d=datetime.fromisoformat(v)
            except ValueError:d=pd.to_datetime(v,dayfirst=True,errors='raise').to_pydatetime()
        else:return None
        if pd.isna(d):return None
        return d.astimezone(LOCAL).replace(tzinfo=None) if d.tzinfo else d
    except (ValueError,TypeError,OverflowError):return None

def headers(rows):
    try:header=next(rows)
    except StopIteration:raise ValueError('A required sheet is empty.')
    cols={key(v):i for i,v in enumerate(header) if text(v)}
    def get(row,*names):
        for name in names:
            i=cols.get(key(name))
            if i is not None and i<len(row) and row[i] is not None:return row[i]
        return None
    return cols,get

def parse_workbook(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        if sum(i.file_size for i in z.infolist())>150*1024*1024:raise ValueError('Workbook is too large to refresh safely (150 MB uncompressed limit).')
    book=load_workbook(io.BytesIO(content),read_only=True,data_only=True)
    return parse_book(book)

def parse_csv_bundle(contents):
    """Use the same validated register/event parser for the public CSV bundle."""
    class Sheet:
        def __init__(self,content):self.values=list(csv.reader(io.StringIO(content.decode('utf-8-sig'))))
    class Book:
        sheetnames=['Equipments','Breakdowns']
        def __init__(self):self.sheets={'Equipments':Sheet(contents['assets.csv']),'Breakdowns':Sheet(contents['events.csv'])}
        def __getitem__(self,name):return self.sheets[name]
        def close(self):pass
    return parse_book(Book())

def parse_book(book):
    try:
        names={key(n):n for n in book.sheetnames}
        if 'equipments' not in names or 'breakdowns' not in names:raise ValueError('Workbook needs Equipments and Breakdowns sheets.')
        rows=iter(book[names['equipments']].values);cols,get=headers(rows)
        if 'equipmentid' not in cols or not ({'equipments','equipment'}&set(cols)) or not ({'stations','station'}&set(cols)):raise ValueError('Equipments needs Equipment ID, Equipments and Stations columns.')
        assets={};warnings=Counter();examples=[]
        for line,row in enumerate(rows,2):
            if line>250000:raise ValueError('Register exceeds the 250,000-row limit.')
            if not any(v is not None for v in row):continue
            eid=text(get(row,'Equipment ID'));name=text(get(row,'Equipments','Equipment'));station=text(get(row,'Stations','Station'))
            if not eid or not name or not station:
                warnings['Incomplete register rows excluded']+=1;continue
            item={'equipment_id':eid,'equipment':name,'station':station,'equipment_class':text(get(row,'Equipment Class'))}
            if eid in assets:
                if assets[eid]!=item:raise ValueError(f'Conflicting register entries for Equipment ID {eid}. Resolve them before refreshing.')
                warnings['Duplicate register IDs collapsed']+=1;continue
            assets[eid]=item
        if not assets:raise ValueError('No valid registered equipment was found.')
        lookup=defaultdict(list)
        for a in assets.values():lookup[(a['station'].casefold(),a['equipment'].casefold())].append(a['equipment_id'])
        rows=iter(book[names['breakdowns']].values);cols,get=headers(rows)
        if 'maintenancetype' not in cols:raise ValueError('Breakdowns needs a Maintenance Type column. Work categories must be explicit.')
        if 'equipmentid' not in cols and not ({'equipments','equipment'}&set(cols)):raise ValueError('Breakdowns needs Equipment ID or equipment/station names.')
        events=[];seen={};raw_count=0
        for line,row in enumerate(rows,2):
            if line>250000:raise ValueError('Breakdowns exceeds the 250,000-row limit.')
            if not any(v is not None for v in row):continue
            raw_count+=1;eid=text(get(row,'Equipment ID'));name=text(get(row,'Equipments','Equipment'));station=text(get(row,'Stations','Station'))
            if not eid:
                matches=lookup.get((station.casefold(),name.casefold()),[])
                if len(matches)==1:eid=matches[0];warnings['Event IDs resolved from exact station/name']+=1
            if eid not in assets:
                warnings['Unregistered or ambiguous equipment rows excluded']+=1
                if len(examples)<8:examples.append(f'Breakdowns row {line}: {eid or name or "missing equipment"}')
                continue
            a=assets[eid]
            if station and station!=a['station']:warnings['Event station disagrees with register; register used']+=1
            kind=text(get(row,'Maintenance Type')).lower().replace('_','-').replace('–','-')
            for t in TYPES:
                if kind.startswith(t+'(') or kind.startswith(t+' ('):kind=t
            if kind not in TYPES:kind='Unclassified';warnings['Work type missing or unrecognized']+=1
            start=stamp(get(row,'Maintenance Start Timestamp'))
            if start is None:
                start=stamp(get(row,'Date'))
                tm=get(row,'Start Time')
                if start and tm is not None:
                    if isinstance(tm,time):start=datetime.combine(start.date(),tm)
                    else:
                        try:start=datetime.combine(start.date(),time.fromisoformat(str(tm)))
                        except (ValueError,TypeError):pass
            finish=stamp(get(row,'Maintenance End Timestamp'))
            down_raw=get(row,'Operational Downtime (hrs)');down=numeric(down_raw)
            if kind=='breakdown' and down is None:warnings['Breakdown downtime missing or invalid']+=1
            if kind=='breakdown' and start is None:warnings['Undated breakdowns']+=1
            source=text(get(row,'Event Record Source')) or 'Unspecified'
            failure=text(get(row,'Failure Event ID'));work=text(get(row,'Work Order ID'));fault=text(get(row,'Fault Description'))
            identity=('work',work) if work else ('row',failure,kind,str(start),str(finish),fault,str(down))
            identity=(eid,)+identity
            entry={**a,'maintenance_type':kind,'started_at':start.isoformat() if start else None,'ended_at':finish.isoformat() if finish else None,'downtime_hours':down,'record_source':source,'failure_event_id':failure,'work_order_id':work,'fault':fault,'failure_mode':text(get(row,'Failure Mode Code')) or text(get(row,'Component')) or 'Not recorded','workbook_row':line}
            comparable={k:v for k,v in entry.items() if k!='workbook_row'}
            if identity in seen:
                if seen[identity]!=comparable:raise ValueError(f'Conflicting work order ID for {eid} at Breakdowns row {line}. Resolve duplicate IDs before refreshing.')
                warnings['Duplicate event rows collapsed']+=1;continue
            seen[identity]=comparable;events.append(entry)
        work_records=events;groups=defaultdict(list);non_breakdowns=[]
        for e in work_records:
            if e['maintenance_type']=='breakdown':groups[(e['equipment_id'],e['failure_event_id'] or e['work_order_id'] or ('row-'+str(e['workbook_row'])))].append(e)
            else:non_breakdowns.append(e)
        episodes=[]
        for group in groups.values():
            group.sort(key=lambda e:(e['started_at'] is None,e['started_at'] or '',e['workbook_row']))
            episode=dict(group[0]);episode['workbook_rows']=[e['workbook_row'] for e in group];episode['linked_work_records']=len(group)
            if len(group)>1:
                warnings['Follow-up work records grouped into existing failure events']+=len(group)-1
                starts=[e['started_at'] for e in group if e['started_at']];ends=[e['ended_at'] for e in group if e['ended_at']]
                episode['started_at']=min(starts) if starts else None;episode['ended_at']=max(ends) if ends else None
                values=[e['downtime_hours'] for e in group];episode['downtime_hours']=sum(values) if all(v is not None for v in values) else None
                episode['work_order_id']=', '.join(dict.fromkeys(e['work_order_id'] for e in group if e['work_order_id']))
                episode['fault']='; '.join(dict.fromkeys(e['fault'] for e in group if e['fault']))
                if len({e['record_source'] for e in group})>1:warnings['Linked failure events with mixed sources (earliest source used)']+=1
            episodes.append(episode)
        return {'assets':list(assets.values()),'events':episodes+non_breakdowns,'work_records':work_records,'quality':dict(warnings),'issue_examples':examples,'raw_maintenance_rows':raw_count,'accepted_maintenance_events':len(work_records),'unique_breakdown_events':len(episodes),'sources':sorted({e['record_source'] for e in work_records})}

    finally:book.close()

class WorkbookAnalytics:
    def __init__(self,source_path=None,config_path=None,allowed_root=None,model_sha256=None):
        self.root=Path(allowed_root or APP.parent).resolve();self.config=Path(config_path or APP/'data/analytics_config.json');self.model_sha=model_sha256;self.lock=threading.RLock();self.snapshot=None;self.meta=None;self.signature=None;self.error=None
        default=APP/'data/demo/events.csv'
        self.path=Path(source_path).resolve() if source_path else default.resolve()
        if not source_path and self.config.exists():
            try:
                config=json.loads(self.config.read_text(encoding='utf-8'))
                self.path=self.resolve_path(APP/config['bundled_workbook'] if config.get('bundled_workbook') else config['source_path'])
            except (ValueError,KeyError,OSError,json.JSONDecodeError):self.error='Analytics source configuration could not be read. Select a valid workbook.'
    def resolve_path(self,value):
        p=Path(text(value).strip(chr(34)));p=p if p.is_absolute() else self.root/p;p=p.resolve()
        if not p.is_relative_to(self.root):raise ValueError('Choose an existing workbook inside this maintenance project folder.')
        if (p.suffix.lower()!='.xlsx' and p.name!='events.csv') or p.name.startswith('~$'):raise ValueError('Choose an .xlsx workbook or an events.csv demo bundle.')
        if not p.is_file():raise ValueError('Workbook not found. Check its path and save it in the maintenance project folder.')
        if p.suffix.lower()=='.csv':
            for member in self._bundle_paths(p):
                if not member.resolve().is_relative_to(self.root) or not member.is_file():raise ValueError('The CSV bundle needs assets.csv, events.csv and observations.csv inside this project.')
        return p
    @staticmethod
    def _bundle_paths(path):return [path.with_name(name) for name in ['assets.csv','events.csv','observations.csv']]
    def _signature(self,path):
        paths=self._bundle_paths(path) if path.suffix.lower()=='.csv' else [path]
        stats=[p.stat() for p in paths]
        if sum(s.st_size for s in stats)>40*1024*1024:raise ValueError('Data source exceeds the 40 MB refresh limit.')
        return tuple((s.st_mtime_ns,s.st_size) for s in stats) if len(stats)>1 else (stats[0].st_mtime_ns,stats[0].st_size)
    def _read(self,path):
        if path.suffix.lower()=='.csv':
            before=self._signature(path)
            contents={p.name:p.read_bytes() for p in self._bundle_paths(path)}
            after=self._signature(path)
            if before!=after:raise ValueError('CSV files changed during refresh. Finish saving, then refresh again.')
            data=parse_csv_bundle(contents);digest=demo_source_digest(contents)
            saved=max(p.stat().st_mtime for p in self._bundle_paths(path))
            meta={'path':str(path),'filename':'Synthetic CSV demo bundle','source_format':'csv','sha256':digest,'saved_at':datetime.fromtimestamp(saved,LOCAL).isoformat(timespec='seconds'),'loaded_at':now(),'checked_at':now(),'matches_model_snapshot':digest==self.model_sha,'raw_maintenance_rows':data['raw_maintenance_rows'],'accepted_maintenance_events':data['accepted_maintenance_events'],'registered_equipment':len(data['assets']),'unique_breakdown_events':data['unique_breakdown_events'],'quality':data['quality'],'issue_examples':data['issue_examples']}
            return data,meta,after
        before=path.stat()
        if before.st_size>40*1024*1024:raise ValueError('Workbook exceeds the 40 MB refresh limit.')
        content=path.read_bytes();after=path.stat()
        if (before.st_mtime_ns,before.st_size)!=(after.st_mtime_ns,after.st_size):raise ValueError('Excel was saving during refresh. Wait for the save to finish, then refresh again.')
        digest=hashlib.sha256(content).hexdigest();data=parse_workbook(content)
        meta={'path':str(path),'filename':path.name,'sha256':digest,'saved_at':datetime.fromtimestamp(after.st_mtime,LOCAL).isoformat(timespec='seconds'),'loaded_at':now(),'checked_at':now(),'matches_model_snapshot':digest==self.model_sha,'raw_maintenance_rows':data['raw_maintenance_rows'],'accepted_maintenance_events':data['accepted_maintenance_events'],'registered_equipment':len(data['assets']),'unique_breakdown_events':data['unique_breakdown_events'],'quality':data['quality'],'issue_examples':data['issue_examples']}
        return data,meta,(after.st_mtime_ns,after.st_size)
    def refresh(self,force=False):
        with self.lock:
            try:
                signature=self._signature(self.path)
                if force or self.snapshot is None or signature!=self.signature:
                    data,meta,sig=self._read(self.path);self.snapshot,self.meta,self.signature=data,meta,sig
                self.error=None;self.meta['checked_at']=now()
            except Exception as error:
                # A failed import never replaces a valid cached snapshot with partial/zero results.
                self.error=f'Workbook refresh failed: {error}'
                if self.snapshot is None:raise ValueError(self.error) from error
            return self.snapshot,{**self.meta,'attempted_at':now(),'stale':bool(self.error),'refresh_error':self.error}
    def set_source(self,value):
        with self.lock:
            path=self.resolve_path(value)
            try:data,meta,signature=self._read(path)
            except Exception as error:raise ValueError('Workbook was not accepted: '+str(error)) from error
            self.config.parent.mkdir(parents=True,exist_ok=True);tmp=self.config.with_suffix('.tmp');tmp.write_text(json.dumps({'source_path':str(path)},indent=2),encoding='utf-8');tmp.replace(self.config)
            self.path,self.snapshot,self.meta,self.signature,self.error=path,data,meta,signature,None
            return meta
    def summarize(self,query=None,force=False):
        query=query or {};data,meta=self.refresh(force);station=text(query.get('station'));source=text(query.get('record_source')) or 'all'
        if station and station not in {a['station'] for a in data['assets']}:raise ValueError('Select a station in the current workbook.')
        if source!='all' and source not in data['sources']:raise ValueError('Select a record source in the current workbook.')
        try:top=int(query.get('top_n',10))
        except (TypeError,ValueError):raise ValueError('Choose top 5, 10 or 20.')
        if top not in [5,10,20]:raise ValueError('Choose top 5, 10 or 20.')
        rank=text(query.get('rank_by')) or 'breakdowns'
        if rank not in ['breakdowns','downtime']:raise ValueError('Choose breakdown count or recorded downtime.')
        dates={}
        for name in ['start','end']:
            value=text(query.get(name))
            try:dates[name]=date.fromisoformat(value) if value else None
            except ValueError:raise ValueError('Use YYYY-MM-DD for date filters.')
        if dates['start'] and dates['end'] and dates['start']>dates['end']:raise ValueError('Start date must be on or before end date.')
        def eligible(e):
            if station and e['station']!=station:return False
            if source!='all' and e['record_source']!=source:return False
            d=date.fromisoformat(e['started_at'][:10]) if e['started_at'] else None
            return not ((dates['start'] and (d is None or d<dates['start'])) or (dates['end'] and (d is None or d>dates['end'])))
        events=[e for e in data['events'] if eligible(e)];work_records=[e for e in data['work_records'] if eligible(e)];failures=[e for e in events if e['maintenance_type']=='breakdown'];assets=[a for a in data['assets'] if not station or a['station']==station];byid=defaultdict(list)
        for e in failures:byid[e['equipment_id']].append(e)
        def stats(records):
            values=[e['downtime_hours'] for e in records if e['downtime_hours'] is not None]
            return {'breakdowns':len(records),'recorded_downtime_hours':round(sum(values),3) if values else None,'downtime_known_events':len(values),'mean_recorded_downtime_hours':round(sum(values)/len(values),3) if values else None}
        equipment=[{**a,**stats(byid.get(a['equipment_id'],[]))} for a in assets]
        order=lambda a:(-a['breakdowns'],a['equipment_id']) if rank=='breakdowns' else (-(a['recorded_downtime_hours'] if a['recorded_downtime_hours'] is not None else -1),-a['breakdowns'],a['equipment_id'])
        ranked=sorted([e for e in equipment if e['breakdowns']>0],key=order);top_rows=ranked[:top]
        station_rows=[]
        for name in sorted({a['station'] for a in assets}):
            group=[a for a in equipment if a['station']==name];rs=[e for e in failures if e['station']==name];n=len(group)
            station_rows.append({'station':name,'registered_equipment':n,'affected_equipment':sum(a['breakdowns']>0 for a in group),'zero_recorded_breakdown_equipment':sum(a['breakdowns']==0 for a in group),'breakdowns_per_equipment':len(rs)/n if n else None,**stats(rs)})
        station_rows.sort(key=lambda s:(-s['breakdowns'],s['station']))
        dated=[e for e in events if e['started_at']];all_dates=[e['started_at'][:10] for e in data['events'] if e['started_at']]
        first=dates['start'] or (date.fromisoformat(min(e['started_at'][:10] for e in dated)) if dated else None);last=dates['end'] or (date.fromisoformat(max(e['started_at'][:10] for e in dated)) if dated else None)
        months=[]
        if first and last:
            d=first.replace(day=1);end_month=last.replace(day=1)
            while d<=end_month:
                if len(months)>1200:raise ValueError('Choose a reporting period of at most 100 years.')
                month=d.strftime('%Y-%m');rs=[e for e in failures if e['started_at'] and e['started_at'].startswith(month)];months.append({'month':month,**stats(rs),'stations':{s['station']:sum(e['station']==s['station'] for e in rs) for s in station_rows}})
                d=date(d.year+1,1,1) if d.month==12 else date(d.year,d.month+1,1)
        cause=Counter(e['failure_mode'] for e in failures);cause_rows=[];cumulative=0
        for name,count in sorted(cause.items(),key=lambda x:(-x[1],x[0]))[:10]:
            cumulative+=count;cause_rows.append({'cause':name,'breakdowns':count,'share':count/len(failures) if failures else None,'cumulative_share':cumulative/len(failures) if failures else None})
        counts=Counter(e['maintenance_type'] for e in work_records);mix=[{'maintenance_type':t,'records':counts[t]} for t in TYPES+['Unclassified'] if counts[t]]
        station_min=min((s['breakdowns'] for s in station_rows),default=None);normalized_min=min((s['breakdowns_per_equipment'] for s in station_rows),default=None)
        summary={**stats(failures),'maintenance_events':len(work_records),'registered_equipment':len(assets),'affected_equipment':len(byid),'repeat_breakdown_equipment':sum(len(v)>=2 for v in byid.values()),'zero_recorded_breakdown_equipment':sum(a['breakdowns']==0 for a in equipment),'top20_breakdown_share':sum(a['breakdowns'] for a in sorted(equipment,key=lambda a:-a['breakdowns'])[:20])/len(failures) if failures else None,'synthetic_breakdowns':sum(e['record_source'].casefold().startswith('synthetic') for e in failures),'undated_breakdowns_in_scope':sum(not e['started_at'] for e in failures),'most_breakdowns_stations':[s['station'] for s in station_rows if s['breakdowns']==(station_rows[0]['breakdowns'] if station_rows else None)] if failures else [],'fewest_breakdowns_stations':[s['station'] for s in station_rows if s['breakdowns']==station_min] if failures else [],'lowest_per_equipment_stations':[s['station'] for s in station_rows if s['breakdowns_per_equipment']==normalized_min] if failures else []}
        response={'source':meta,'filters':{'station':station,'record_source':source,'start':text(query.get('start')),'end':text(query.get('end')),'top_n':top,'rank_by':rank},'options':{'stations':sorted({a['station'] for a in data['assets']}),'record_sources':data['sources'],'first_event_date':min(all_dates) if all_dates else None,'last_event_date':max(all_dates) if all_dates else None},'summary':summary,'stations':station_rows,'equipment_rankings':top_rows,'equipment_totals':sorted(equipment,key=order),'monthly':months,'causes':cause_rows,'maintenance_mix':mix,'definitions':{'breakdowns':'Distinct per-equipment failure episodes, grouped by Failure Event ID (then Work Order ID or exact row content). Linked follow-up work orders count as one breakdown, assigned to the earliest start date. Other work types are excluded. Work-mix counts remain separate work records.','station_comparison':'Per-equipment values divide selected-period breakdowns by the current registered equipment count, including equipment with no logged breakdown. This does not adjust for operating hours or historical register changes.','downtime':'Operational Downtime (hrs) is summed across distinct linked work orders for each selected failure episode; all linked hours must be known for its total. Events are selected by earliest start date; their full recorded event downtime is included. Maintenance duration is not substituted. Missing downtime remains unavailable. Event totals can overlap and are not plant outage time or availability.','coverage':'No logged breakdown does not prove failure-free operation. Rankings depend on reporting coverage; synthetic quotas do not measure field reliability.','refresh':'Saved workbook changes are checked every minute while Analytics is open. Refresh from Excel forces a read. Charts update independently of the published prediction model.'}}
        return response,failures
