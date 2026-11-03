import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import create_app
from services import MODEL,ml
import unittest
import tempfile
import json
import re
import io
import pandas as pd
import numpy as np

class MaintenanceAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_root=Path(__file__).resolve().parent/'.test_tmp';cls.test_root.mkdir(exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(dir=cls.test_root);cls.app=create_app(Path(cls.temp.name)/'records.sqlite3');cls.app.testing=True
        cls.client=cls.app.test_client();html=cls.client.get('/').get_data(as_text=True)
        cls.token=re.search(r'name="app-token" content="([^"]+)"',html).group(1)
        cls.headers={'X-App-Token':cls.token}
        cls.sample=pd.read_csv(MODEL/'historical_example_input.csv')
    @classmethod
    def tearDownClass(cls):
        assert Path(cls.temp.name).resolve().is_relative_to(cls.test_root.resolve())
        cls.temp.cleanup()
    def observation(self,index=0):return json.loads(self.sample.iloc[[index]].to_json(orient='records'))[0]
    def post(self,path,payload):return self.client.post(path,json=payload,headers=self.headers)
    def test_01_home_overview_and_assets(self):
        r=self.client.get('/api/overview');self.assertEqual(r.status_code,200);self.assertEqual(r.json['counts']['assets'],48)
        expected=int((self.app.extensions['maintenance'].latest['Seven-day failure probability']>=r.json['manifest']['alert_threshold']).sum())
        self.assertEqual(r.json['counts']['alerts'],expected);self.assertEqual(len(r.json['stations']),4)
        self.assertEqual(sum(v['Seven-day failure probability'] is not None and v['Seven-day failure probability']>=r.json['manifest']['alert_threshold'] for v in r.json['equipment']),expected)
        probabilities=[v['Seven-day failure probability'] for v in r.json['equipment'] if v['Seven-day failure probability'] is not None]
        self.assertEqual(probabilities,sorted(probabilities,reverse=True))
        self.assertEqual(self.client.get('/api/equipment/DEMO-A-001').status_code,200)
        self.assertEqual(self.client.get('/api/equipment/UNKNOWN').status_code,400)
    def test_02_exact_historical_model_replay(self):
        service=self.app.extensions['maintenance'];result=service.score(self.sample,mode='historical')
        actual=pd.DataFrame(result).set_index(ml.ID)['Seven-day failure probability']
        expected=service.latest.set_index(ml.ID)['Seven-day failure probability']
        np.testing.assert_allclose(actual.loc[expected.index],expected,atol=1e-12)
        self.assertEqual(len(service.observations()),0)
    def test_03_historical_and_new_modes_are_separate(self):
        row=self.observation();r=self.post('/api/predict',{'observation':row,'mode':'new','save':True});self.assertEqual(r.status_code,400)
        r=self.post('/api/predict',{'observation':row,'mode':'historical','save':True});self.assertEqual(r.status_code,400)
        r=self.post('/api/predict',{'observation':row,'mode':'historical','save':False});self.assertEqual(r.status_code,200);self.assertEqual(r.json['saved'],False)
    def test_04_valid_new_reading_persists(self):
        row=self.observation();row[ml.TIME]='2026-10-06 06:00:00'
        r=self.post('/api/predict',{'observation':row,'mode':'new','save':True});self.assertEqual(r.status_code,200,r.json)
        self.assertEqual(len(self.app.extensions['maintenance'].observations()),1)
        r=self.post('/api/predict',{'observation':row,'mode':'new','save':True});self.assertEqual(r.status_code,400)
    def test_05_bad_inputs_rejected_without_saving(self):
        for field,value in [('Operating Hours Last 24h',25),('Inspection Score (0-100)',101),('Prior Synthetic Breakdowns 30d',1.5),('Station','Wrong station'),('Operating Status','Unknown')]:
            row=self.observation();row[field]=value;r=self.post('/api/predict',{'observation':row,'mode':'historical'});self.assertEqual(r.status_code,400,(field,r.json))
        self.assertEqual(len(self.app.extensions['maintenance'].observations()),1)
    def test_06_idle_not_scored_and_future_labels_ignored(self):
        row=self.observation();row['Operating Status']='Idle';row['Operating Hours Last 24h']=0
        r=self.post('/api/predict',{'observation':row,'mode':'historical'});self.assertEqual(r.status_code,200,r.json)
        self.assertIsNone(r.json['results'][0]['Seven-day failure probability'])
        row=self.observation();first=self.post('/api/predict',{'observation':row,'mode':'historical'}).json
        row[ml.TARGET]='POISON';row['Next Failure Lead Time (hrs)']=-1;second=self.post('/api/predict',{'observation':row,'mode':'historical'}).json
        self.assertEqual(first,second)
    def test_07_batch_is_atomic(self):
        frame=self.sample.iloc[1:3].copy();frame[ml.TIME]='2026-10-06 06:00:00';frame.iloc[1,frame.columns.get_loc('Operating Hours Last 24h')]=30
        r=self.client.post('/api/predict-batch',data={'file':(io.BytesIO(frame.to_csv(index=False).encode()),'readings.csv'),'mode':'new','save':'true'},headers=self.headers)
        self.assertEqual(r.status_code,400);self.assertEqual(len(self.app.extensions['maintenance'].observations()),1)
        r=self.client.post('/api/predict-batch',data={'file':(io.BytesIO(b'not csv'),'file.html')},headers=self.headers);self.assertEqual(r.status_code,400)
    def test_08_categories_and_record_persistence(self):
        base={'equipment_id':'DEMO-A-001','occurred_at':'2026-10-06 08:00:00','maintenance_type':'breakdown','operation_stopped':False,'downtime_hours':0,'notes':'Verified test fixture work'}
        self.assertEqual(self.post('/api/events',base).status_code,400)
        base['operation_stopped']=True;base['downtime_hours']=2.5;base['notes']='=HYPERLINK("malicious","text")'
        r=self.post('/api/events',base);self.assertEqual(r.status_code,201,r.json)
        self.assertEqual(len(self.client.get('/api/events').json['events']),1)
        self.assertIn("'=HYPERLINK",self.client.get('/download/events').get_data(as_text=True))
        base['maintenance_type']='condition-based';self.assertEqual(self.post('/api/events',base).status_code,400)
        base['maintenance_type']='invalid';self.assertEqual(self.post('/api/events',base).status_code,400)
    def test_09_review_queue_is_persistent(self):
        r=self.post('/api/queue',{'equipment_id':'DEMO-B-001','status':'Inspection planned','notes':'Test fixture: inspect bearings'});self.assertEqual(r.status_code,200)
        service=self.app.extensions['maintenance'];fresh=type(service)(service.db_path)
        self.assertEqual(fresh.asset('DEMO-B-001')['queue']['status'],'Inspection planned')
    def test_10_local_write_and_download_rules(self):
        self.assertEqual(self.client.post('/api/queue',json={}).status_code,403)
        self.assertEqual(self.client.get('/api/overview',headers={'Host':'external.example'}).status_code,403)
        response=self.client.get('/download/template');self.assertEqual(response.status_code,200);response.close()
        import io,zipfile
        response=self.client.get('/download/training-workbook');self.assertEqual(response.status_code,200)
        with zipfile.ZipFile(io.BytesIO(response.get_data())) as archive:
            self.assertEqual(set(archive.namelist()),{'assets.csv','events.csv','observations.csv','provenance.json'})
            self.assertFalse(any(name.endswith('.xlsx') for name in archive.namelist()))
        response.close()
        self.assertEqual(self.client.get('/download/unknown').status_code,404)
        self.assertEqual(self.client.get('/').headers['X-Frame-Options'],'DENY')

    def test_11_equipment_selection_uses_saved_dated_reading(self):
        result=self.client.get('/api/forecast/DEMO-B-001').json
        self.assertEqual(result['equipment_id'],'DEMO-B-001');self.assertEqual(result['as_of'],'2026-10-05 06:00:00')
        self.assertTrue(result['stale']);self.assertFalse(result['current_forecast_available']);self.assertIsNone(result['review_plan']['review_by'])
        p=[r['probability'] for r in result['risks']];self.assertEqual(len(p),3);self.assertTrue(0<=p[0]<=p[1]<=p[2]<=1)
        self.assertEqual(result['observed_cycle']['breakdown_count'],0)
    def test_12_only_recall_strategy_is_active(self):
        default=self.client.get('/api/forecast/DEMO-B-001').json
        chosen=self.client.get('/api/forecast/DEMO-B-001?policy=catch_more').json
        self.assertEqual(default['risks'],chosen['risks']);self.assertEqual(default['policy'],'catch_more')
        overview=self.client.get('/api/overview').json
        self.assertEqual(list(overview['forecast_options']['policies']),['catch_more'])
        self.assertTrue(all(list(v['policies'])==['catch_more'] for v in overview['forecast_evaluation']['horizons'].values()))
        self.assertEqual(overview['manifest']['alert_threshold'],chosen['risks'][0]['threshold'])
        for policy in ['balanced','fewer_inspections','invalid']:
            self.assertEqual(self.client.get('/api/forecast/DEMO-B-001?policy='+policy).status_code,400)
            self.assertEqual(self.post('/api/predict',{'observation':self.observation(),'mode':'historical','policy':policy}).status_code,400)
        self.assertEqual(self.client.get('/api/forecast/DEMO-B-001?horizon=365').status_code,400)
    def test_13_manual_thirty_day_scoring_and_horizon_coherence(self):
        result=self.post('/api/predict',{'observation':self.observation(2),'mode':'historical','horizon':30,'policy':'catch_more'})
        self.assertEqual(result.status_code,200,result.json);r=result.json['results'][0]
        self.assertEqual(r['Selected horizon days'],30);self.assertAlmostEqual(r['Selected horizon failure probability'],r['horizon_risks'][2]['probability'],places=12)
        self.assertEqual(pd.Timestamp(r['Forecast horizon ends'])-pd.Timestamp(r[ml.TIME]),pd.Timedelta(days=30))
    def test_14_cycle_requires_intervals_and_excludes_future_events(self):
        import forecasting as fc
        events=pd.DataFrame([{ml.ID:'X','Maintenance Type':'breakdown','Maintenance Start Timestamp':t} for t in ['2026-01-01','2026-01-11','2026-01-21','2026-03-01']])
        result=fc.cycle_summary(events,'X',pd.Timestamp('2026-01-25'))
        self.assertEqual(result['breakdown_count'],3);self.assertEqual(result['median_interval_days'],10);self.assertIsNone(result['predicted_failure_date'])
        result=fc.cycle_summary(events,'X',pd.Timestamp('2026-01-15'));self.assertIsNone(result['median_interval_days'])
    def test_15_forecasting_does_not_save_records(self):
        service=self.app.extensions['maintenance'];before=(len(service.observations()),len(service.events()))
        for h in [7,14,30]:self.assertEqual(self.client.get('/api/forecast/DEMO-C-001?horizon='+str(h)).status_code,200)
        self.assertEqual(before,(len(service.observations()),len(service.events())))
    def test_16_new_reading_enables_dated_review_and_later_work_requires_refresh(self):
        from datetime import datetime,timedelta
        from zoneinfo import ZoneInfo
        row=self.observation(3);now=datetime.now(ZoneInfo('Africa/Lagos')).replace(tzinfo=None,microsecond=0)
        row[ml.TIME]=str(now-timedelta(minutes=2));eid=row[ml.ID]
        result=self.post('/api/predict',{'observation':row,'mode':'new','save':True});self.assertEqual(result.status_code,200,result.json)
        result=self.client.get('/api/forecast/'+eid).json;self.assertTrue(result['current_forecast_available']);self.assertEqual(result['review_plan']['review_by'] is not None,any(r['inspect'] for r in result['risks']))
        event={'equipment_id':eid,'occurred_at':str(now-timedelta(minutes=1)),'maintenance_type':'inspection','operation_stopped':False,'downtime_hours':0,'notes':'Temporary integration-test inspection fixture'}
        self.assertEqual(self.post('/api/events',event).status_code,201)
        result=self.client.get('/api/forecast/'+eid).json;self.assertTrue(result['work_after_reading']);self.assertFalse(result['current_forecast_available']);self.assertIsNone(result['review_plan']['review_by'])

    def test_17_active_estimator_is_gradient_boosting(self):
        from sklearn.ensemble import GradientBoostingClassifier
        service=self.app.extensions['maintenance'];engine=service.forecasts
        self.assertEqual(engine.manifest['algorithm'],'GradientBoostingClassifier')
        self.assertEqual(self.client.get('/api/health').json['algorithm'],'GradientBoostingClassifier')
        for models in engine.models.values():
            for model in models:self.assertIsInstance(model.named_steps['classifier'],GradientBoostingClassifier)
        result=self.post('/api/predict',{'observation':self.observation(),'mode':'historical'}).json['results'][0]
        self.assertEqual(result['Model algorithm'],'GradientBoostingClassifier')
        self.assertIn('tree-path',result['explanation_method'].lower())
        self.assertTrue(all(d['method']=='tree_path' for d in result['drivers']))

    def test_18_public_demo_provenance_and_report(self):
        overview=self.client.get('/api/overview').json
        self.assertIsNone(overview['optimization_review'])
        self.assertTrue(overview['forecast_evaluation']['synthetic_data'])
        self.assertEqual(overview['forecast_evaluation']['dataset_revision'],'Independent synthetic public demonstration')
        report=self.client.get('/download/report');self.assertEqual(report.status_code,200)
        self.assertIn('Synthetic demonstration evaluation',report.get_data(as_text=True));report.close()
        self.assertEqual(list(overview['forecast_options']['policies']),['catch_more'])

if __name__=='__main__':unittest.main()
