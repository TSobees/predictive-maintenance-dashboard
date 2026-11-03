"""Create fictional maintenance history from fixed rules; reads no private files."""
from pathlib import Path
import argparse
import json
import random
import csv
from datetime import datetime,timedelta
import math

ROOT=Path(__file__).resolve().parents[1]
SEED=314159
START=datetime(2025,10,6,6)
DAYS=365

def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def generate(out):
    rng=random.Random(SEED);assets=[];observations=[];events=[]
    classes=['Pump','Fan','Motor','Conveyor','Compressor','Mixer']
    for si,letter in enumerate('ABCD'):
        for number in range(1,13):
            eid=f'DEMO-{letter}-{number:03d}';kind=classes[(number-1)%len(classes)]
            name=f'Demo {kind} {letter}{number:02d}';station=f'Demo Station {letter}'
            typical=12+number%5;criticality=1+number%5
            assets.append({'Equipment ID':eid,'Equipments':name,'Stations':station,'Equipment Class':kind,
                           'Simulated Criticality (1-5)':criticality,'Simulated Typical Daily Hours':typical,
                           'Record Source':'Synthetic public demonstration'})
            health=rng.uniform(.1,.5);meter=0.;last_maintenance=START-timedelta(days=15)
            failures=[];next_downtime=0;event_index=0
            def work(when,category,downtime,fault,mode,component,event_id=None):
                nonlocal event_index
                event_index+=1
                event={'Equipment ID':eid,'Equipments':name,'Stations':station,'Maintenance Type':category,
                       'Maintenance Start Timestamp':when.strftime('%Y-%m-%d %H:%M:%S'),'Maintenance End Timestamp':(when+timedelta(hours=max(downtime,0.5))).strftime('%Y-%m-%d %H:%M:%S'),
                       'Operational Downtime (hrs)':round(downtime,2),'Failure Event ID':event_id or '',
                       'Work Order ID':f'DEMO-WO-{letter}{number:02d}-{event_index:04d}',
                       'Event Record Source':'Synthetic public demonstration','Fault Description':fault,
                       'Action Taken':'Simulated work for software demonstration','Failure Mode Code':mode,
                       'Component':component,'Operating State At Work':'Out of service' if category=='refurbishment' else 'Stopped' if category=='breakdown' else 'Running'}
                events.append(event)
            for day in range(DAYS):
                t=START+timedelta(days=day)
                daily=max(0.,min(24.,typical+rng.gauss(0,1)-next_downtime));next_downtime=0;meter+=daily
                # Sample condition before that day's future random event.
                score=max(0,min(100,health*100+rng.gauss(0,8)))
                if day%7==0:
                    status='Degraded' if score>=65 else 'Running'
                    if letter=='D' and number==12:status='Idle'
                    observations.append({'Equipment ID':eid,'Equipment':name,'Station':station,
                        'Observation Timestamp':str(t),'Operating Status':status,
                        'Operating Hours Last 24h':round(daily if status!='Idle' else 0,2),
                        'Cumulative Operating Hours':round(meter,2),
                        'Hours Since Last Synthetic Maintenance':round((t-last_maintenance).total_seconds()/3600,2),
                        'Prior Synthetic Breakdowns 30d':sum(t-timedelta(days=30)<f<=t for f in failures),
                        'Inspection Score (0-100)':round(score,2),'Equipment Age (days)':100+number*10+day,
                        'Bearing Temperature (C)':round(35+health*40+rng.gauss(0,3),2),
                        'Vibration RMS (mm/s)':round(max(.1,1+health*7+rng.gauss(0,.6)),2),
                        'Motor Load (%)':round(min(145,max(0,45+health*40+rng.gauss(0,7))),2),
                        'Pressure (bar)':round(max(0,5-health+rng.gauss(0,.4)),2) if kind in ['Pump','Compressor'] else None,
                        'Record Source':'Synthetic public demonstration'})
                if letter=='D' and number==12:continue
                hazard=.001+.025/(1+math.exp(-(health-.7)*8))
                if rng.random()<hazard:
                    when=t+timedelta(hours=3);failures.append(when)
                    failure_id=f'DEMO-FE-{letter}{number:02d}-{len(failures):03d}'
                    dt=rng.uniform(2,10);mode=rng.choice(['BEARING','DRIVE','SEAL','ELECTRICAL'])
                    work(when,'breakdown',dt,'Simulated unplanned stop',mode,kind,event_id=failure_id)
                    # One deliberate follow-up example tests episode de-duplication.
                    if eid=='DEMO-A-005' and len(failures)==1:
                        work(when+timedelta(hours=dt+.5),'breakdown',1.5,'Simulated follow-up repair',mode,kind,event_id=failure_id)
                    next_downtime=dt;health=rng.uniform(.05,.25);last_maintenance=when+timedelta(hours=dt)
                elif day%30==15:
                    category=['inspection','lubrication','condition-based','support','modification','refurbishment'][(day//30+number)%6]
                    work(t+timedelta(hours=3),category,0,'Simulated planned activity','PLANNED',kind)
                    if category in ['lubrication','refurbishment']:health=max(.02,health-.15)
                    last_maintenance=t+timedelta(hours=4)
                health=min(1.3,health+rng.uniform(.004,.02)*(1+si*.08))
    # Deterministic row shuffling prevents station-sorted charts from relying on source order.
    rng.shuffle(events)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('assets.csv',assets),('observations.csv',observations),('events.csv',events)]:write_csv(out/name,rows)
    info={'seed':SEED,'start':str(START),'observation_end':str(START+timedelta(days=364)),
          'coverage_end':str(START+timedelta(days=DAYS)), 'equipment_count':len(assets),
          'observation_count':len(observations),'work_records':len(events),
          'provenance':'Independent fictional simulation. No workbook, company records or original equipment register used.',
          'label_definition':'At least one distinct breakdown episode strictly after observation and within horizon; incomplete horizons are excluded.',
          'license':'MIT','warning':'Simulation performance cannot establish field performance.'}
    (out/'provenance.json').write_text(json.dumps(info,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'maintenance_app/data/demo')
    generate(p.parse_args().output)
