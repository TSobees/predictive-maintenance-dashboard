import sys,json,re,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import create_app
from planning import local_now
from services import ml,MODEL
import pandas as pd

class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).parent/'.test_tmp';cls.root.mkdir(exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(dir=cls.root);cls.app=create_app(Path(cls.temp.name)/'plans.sqlite3');cls.app.testing=True
        cls.service=cls.app.extensions['maintenance'];cls.client=cls.app.test_client()
        cls.headers={'X-App-Token':re.search(r'name="app-token" content="([^"]+)"',cls.client.get('/').text).group(1)}
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def setUp(self):
        with self.service.connect() as db:
            for t in ['maintenance_completions','maintenance_plans','observations','maintenance_events','inspection_queue']:db.execute('DELETE FROM '+t)
        self.now=local_now().floor('s')
    def post(self,url,payload):return self.client.post(url,json=payload,headers=self.headers)
    def plan(self,**changes):
        p={'equipment_id':'DEMO-A-001','maintenance_type':'lubrication','task':'Lubricate main bearings','due_at':str(self.now-pd.DateOffset(days=1)),'interval_days':30};p.update(changes)
        r=self.post('/api/plans',p);self.assertEqual(r.status_code,201,r.json);return r.json['plan']
    def done(self,p,**changes):
        x={'revision':p['revision'],'completed_at':str(self.now-pd.DateOffset(minutes=1)),'operation_stopped':False,'downtime_hours':0,'notes':'Lubricated bearings with approved lubricant'};x.update(changes)
        return self.post('/api/plans/'+p['id']+'/complete',x)
    def test_calendar_alerts_are_independent_of_model_and_review(self):
        self.plan();d=self.service.maintenance_status('DEMO-A-001',now=self.now)
        self.assertEqual(d['plans'][0]['status'],'Overdue');self.assertEqual(d['summary']['overdue_tasks'],1)
        self.assertEqual(d['summary']['model_inspection_flags'],0);self.assertTrue(d['equipment'][0]['reading_required'])
        self.post('/api/queue',{'equipment_id':'DEMO-A-001','status':'Reviewed','notes':'An acknowledgement is not a completed task'})
        self.assertEqual(self.service.maintenance_status('DEMO-A-001')['plans'][0]['status'],'Overdue')
        self.plan(due_at=str(self.now+pd.DateOffset(days=3)));self.plan(due_at=str(self.now+pd.DateOffset(days=20)))
        d=self.service.maintenance_status('DEMO-A-001',now=self.now);self.assertEqual(d['summary']['due_soon_tasks'],1);self.assertEqual(d['summary']['active_plans'],3)
    def test_completion_is_atomic_recurs_and_rejects_duplicate(self):
        p=self.plan();other=self.plan(task='Separate monthly inspection',maintenance_type='inspection')
        r=self.done(p);self.assertEqual(r.status_code,201,r.json)
        self.assertEqual(pd.Timestamp(r.json['completion']['next_due_at']),self.now-pd.DateOffset(minutes=1)+pd.DateOffset(days=30))
        self.assertEqual(len(self.service.events()),1);self.assertEqual(self.done(p).status_code,400)
        plans={p['id']:p for p in self.service.plans()};self.assertEqual(plans[p['id']]['revision'],2);self.assertEqual(plans[other['id']]['revision'],1)
        status=self.service.maintenance_status('DEMO-A-001');self.assertEqual(status['summary']['overdue_tasks'],1)
    def test_invalid_completion_leaves_overdue_task_untouched(self):
        p=self.plan()
        for change in [{'completed_at':str(self.now+pd.DateOffset(days=1))},{'notes':'bad'},{'downtime_hours':1}]:self.assertEqual(self.done(p,**change).status_code,400)
        self.assertEqual(len(self.service.events()),0);self.assertEqual(self.service.plans()[0]['revision'],1)
    def test_one_time_task_closes_and_persists(self):
        p=self.plan(interval_days=None);self.assertEqual(self.done(p).status_code,201)
        self.assertFalse(self.service.plans()[0]['enabled'])
        self.service.init_planning();self.assertFalse(self.service.plans()[0]['enabled']);self.assertEqual(self.service.maintenance_status('DEMO-A-001')['summary']['overdue_tasks'],0)
    def test_hours_use_actual_readings_and_completion_meter(self):
        p=self.plan(due_at=None,interval_days=None,due_meter=100,interval_hours=250)
        status=self.service.maintenance_status('DEMO-A-001');self.assertEqual(status['plans'][0]['status'],'Reading needed');self.assertIsNone(status['plans'][0]['actual_meter'])
        row=json.loads(pd.read_csv(MODEL/'historical_example_input.csv').query('`Equipment ID` == "DEMO-A-001"').iloc[[0]].to_json(orient='records'))[0]
        row[ml.TIME]=str(self.now-pd.DateOffset(minutes=3));row['Cumulative Operating Hours']=110
        self.assertEqual(self.post('/api/predict',{'observation':row,'mode':'new','save':True}).status_code,200)
        self.assertEqual(self.service.maintenance_status('DEMO-A-001')['plans'][0]['status'],'Overdue')
        self.assertEqual(self.done(p).status_code,400)
        r=self.done(p,completed_meter=115);self.assertEqual(r.status_code,201,r.json);self.assertEqual(r.json['completion']['next_due_meter'],365)
        status=self.service.maintenance_status('DEMO-A-001');self.assertEqual(status['plans'][0]['actual_meter'],115);self.assertEqual(status['plans'][0]['status'],'Scheduled')
        self.assertTrue(status['equipment'][0]['reading_required'])
    def test_bad_plans_and_stale_edits_are_rejected(self):
        for change in [{'equipment_id':'BAD'},{'due_at':None},{'interval_days':0},{'interval_days':2.5},{'due_at':'garbage'},{'due_meter':-1}]:
            payload={'equipment_id':'DEMO-A-001','maintenance_type':'inspection','task':'Inspect bearings','due_at':str(self.now)};payload.update(change)
            self.assertEqual(self.post('/api/plans',payload).status_code,400,change)
        p=self.plan();edit={**p,'task':'Updated bearing task','revision':p['revision']};self.assertEqual(self.post('/api/plans',edit).status_code,201)
        self.assertEqual(self.post('/api/plans',edit).status_code,400)
    def test_status_read_has_no_writes_and_export_escapes_formulas(self):
        p=self.plan(task='=HYPERLINK("url","task")')
        d=self.client.get('/api/maintenance-status?equipment_id=DEMO-A-001');self.assertEqual(d.status_code,200,d.json)
        self.assertEqual(len(self.service.events()),0);self.assertEqual(len(self.service.observations()),0)
        self.assertIn("'=HYPERLINK",self.client.get('/download/plans').text)
        self.assertEqual(self.client.post('/api/plans',json={}).status_code,403)
    def test_completion_time_and_downtime_do_not_create_future_work(self):
        p=self.plan(maintenance_type='breakdown')
        r=self.done(p,operation_stopped=True,downtime_hours=2)
        self.assertEqual(r.status_code,201,r.json)
        event=self.service.events()[0]
        self.assertEqual(pd.Timestamp(event['occurred_at'])+pd.DateOffset(hours=2),pd.Timestamp(r.json['completion']['completed_at']))
        self.assertLessEqual(pd.Timestamp(event['occurred_at'])+pd.DateOffset(hours=2),self.now)

    def test_due_boundaries_and_stale_meter(self):
        p=self.plan(due_at=str(self.now),interval_days=None)
        self.assertEqual(self.service.maintenance_status('DEMO-A-001',now=self.now)['plans'][0]['status'],'Overdue')
        self.assertEqual(self.service.maintenance_status('DEMO-A-001',now=self.now-pd.DateOffset(days=8))['plans'][0]['status'],'Scheduled')

if __name__=='__main__':unittest.main()
