import io,zipfile,shutil
"""Analytics semantics and real workbook save/refresh integration. Fixtures never edit user workbooks."""
from pathlib import Path
import sys,unittest,tempfile,shutil,json,hashlib,re,copy
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from services import ml
from analytics import WorkbookAnalytics,parse_workbook
from app import create_app

class Sheet:
    def __init__(self,rows):self.rows=rows
    @property
    def values(self):return iter(self.rows)
class Book:
    sheetnames=['Equipments','Breakdowns']
    def __init__(self,register,records):self.sheets={'Equipments':Sheet(register),'Breakdowns':Sheet(records)}
    def __getitem__(self,name):return self.sheets[name]
    def close(self):pass

class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=R/'tests/.test_tmp';cls.root.mkdir(exist_ok=True);cls.temp=tempfile.TemporaryDirectory(dir=cls.root);cls.folder=Path(cls.temp.name)
        cls.new=R/'data/demo/events.csv'
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as archive:archive.writestr('fixture-marker','Unit-test container; parser is mocked.')
        cls.content=buffer.getvalue()
        cls.tracked=cls.folder/'fixture.xlsx';cls.tracked.write_bytes(cls.content)
        cls.bundle=cls.folder/'bundle';cls.bundle.mkdir()
        for name in ['assets.csv','events.csv','observations.csv']:shutil.copyfile(R/'data/demo'/name,cls.bundle/name)
        cls.app=create_app(cls.folder/'records.sqlite3',analytics_source=cls.bundle/'events.csv',analytics_config=cls.folder/'source_config.json',analytics_root=cls.folder);cls.app.testing=True;cls.client=cls.app.test_client()
        cls.token=re.search(r'name="app-token" content="([^"]+)"',cls.client.get('/').text).group(1);cls.headers={'X-App-Token':cls.token}
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def fixture(self):
        register=[['Equipments','Stations','Equipment ID'],['Pump A','Station A','A-1'],['Pump B','Station A','A-2'],['Pump C','Station B','B-1']]
        h=['Equipment ID','Equipments','Stations','Maintenance Type','Maintenance Start Timestamp','Maintenance End Timestamp','Operational Downtime (hrs)','Failure Event ID','Work Order ID','Event Record Source','Fault Description','Failure Mode Code']
        records=[h,['A-1','Pump A','Station A','breakdown','2026-01-31 23:59:59','2026-02-01 02:00:00',2,'F-1','W-1','Original','=HYPERLINK("x","fault")','BEARING'],['A-1','Pump A','Station A','breakdown','2026-02-01 08:00:00','2026-02-01 11:00:00',3,'F-1','W-2','Original','Follow-up repair','BEARING'],['A-1','Pump A','Station A','breakdown','2026-02-20 09:00:00',None,0,'F-2','W-3','Synthetic','Drive trip','DRIVE'],['A-2','Pump B','Station A','breakdown','2026-02-25 09:00:00',None,None,'F-3','W-4','Synthetic','Unknown hours','DRIVE'],['A-2','Pump B','Station A','inspection','2026-02-25 10:00:00',None,0,None,'W-5','Original','Inspection','']]
        records.append(copy.deepcopy(records[1]))
        return register,records
    def parsed(self,register=None,records=None):
        reg,rec=self.fixture()
        with patch('analytics.load_workbook',return_value=Book(register or reg,records or rec)):return parse_workbook(self.content)
    def engine(self,data):
        e=WorkbookAnalytics(self.tracked,allowed_root=self.folder,config_path=self.folder/'fixture_config.json');e.snapshot=data;e.meta={'path':str(self.tracked),'filename':'fixture.xlsx','sha256':'fixture','loaded_at':'2026-10-17T00:00:00+01:00','quality':data['quality'],'issue_examples':[]};st=self.tracked.stat();e.signature=(st.st_mtime_ns,st.st_size);return e
    def test_01_distinct_failure_episodes_and_work_mix(self):
        data=self.parsed();self.assertEqual(len(data['work_records']),5);self.assertEqual(data['unique_breakdown_events'],3)
        d,events=self.engine(data).summarize();self.assertEqual(d['summary']['breakdowns'],3);self.assertEqual(d['summary']['maintenance_events'],5)
        self.assertEqual(d['summary']['recorded_downtime_hours'],5);self.assertEqual(d['summary']['downtime_known_events'],2);self.assertEqual(d['summary']['mean_recorded_downtime_hours'],2.5)
        self.assertEqual(d['summary']['repeat_breakdown_equipment'],1);self.assertEqual(d['summary']['zero_recorded_breakdown_equipment'],1)
        self.assertEqual(d['summary']['affected_equipment'],2)
        self.assertEqual(d['summary']['fewest_breakdowns_stations'],['Station B']);self.assertEqual(d['stations'][0]['breakdowns_per_equipment'],1.5)
        self.assertEqual(data['quality']['Duplicate event rows collapsed'],1);self.assertEqual(data['quality']['Follow-up work records grouped into existing failure events'],1)
    def test_02_end_date_inclusive_and_followups_not_new_failures(self):
        engine=self.engine(self.parsed());d,_=engine.summarize({'start':'2026-01-31','end':'2026-01-31'})
        self.assertEqual(d['summary']['breakdowns'],1);self.assertEqual(d['summary']['recorded_downtime_hours'],5)
        feb,_=engine.summarize({'start':'2026-02-01','end':'2026-02-28'});self.assertEqual(feb['summary']['breakdowns'],2)
        self.assertEqual(sum(m['breakdowns'] for m in feb['monthly']),2)
    def test_03_filters_top_choices_and_missing_values(self):
        engine=self.engine(self.parsed());d,_=engine.summarize({'record_source':'Original','top_n':'5'});self.assertEqual(d['summary']['breakdowns'],1)
        empty,_=engine.summarize({'station':'Station B'});self.assertEqual(empty['summary']['breakdowns'],0);self.assertIsNone(empty['summary']['recorded_downtime_hours']);self.assertFalse(empty['summary']['most_breakdowns_stations'])
        for query in [{'top_n':7},{'rank_by':'bad'},{'station':'wrong'},{'start':'2026-03-01','end':'2026-01-01'}]:
            with self.assertRaises(ValueError):engine.summarize(query)
    def test_04_conflicting_work_ids_and_register_ids_rejected(self):
        reg,rec=self.fixture();rec[-1][6]=99
        with self.assertRaises(ValueError):self.parsed(records=rec)
        reg.append(['Another pump','Station B','A-1'])
        with self.assertRaises(ValueError):self.parsed(register=reg)
    def test_05_unmapped_records_visible_in_quality(self):
        reg,rec=self.fixture();rec.append(['UNKNOWN','Missing pump','Station A','breakdown','2026-01-01',None,2,'BAD','BAD','Original','Unknown asset','UNKNOWN'])
        d=self.parsed(records=rec);self.assertEqual(d['quality']['Unregistered or ambiguous equipment rows excluded'],1);self.assertTrue(d['issue_examples'])
    def test_06_saved_xlsx_change_automatically_updates_counts(self):
        # Isolate cache invalidation from parser tests: simulate a saved XLSX's
        # new bytes and two valid parser results, without a private predecessor.
        first_data=self.parsed();second_data=copy.deepcopy(first_data)
        extra=copy.deepcopy(second_data['events'][0]);extra['failure_event_id']='NEW-DEMO-EVENT'
        extra['work_order_id']='NEW-DEMO-WORK';second_data['events'].append(extra)
        second_data['unique_breakdown_events']+=1
        p=self.folder/'watch.xlsx';p.write_bytes(self.content)
        e=WorkbookAnalytics(p,allowed_root=self.folder,config_path=self.folder/'watch_config.json')
        with patch('analytics.parse_workbook',side_effect=[first_data,second_data]) as parser:
            first,_=e.summarize();e.summarize();self.assertEqual(parser.call_count,1)
            p.write_bytes(self.content+b'changed-save-signature');second,_=e.summarize()
        self.assertEqual(parser.call_count,2)
        self.assertNotEqual(first['source']['sha256'],second['source']['sha256'])
        self.assertEqual(second['summary']['breakdowns'],first['summary']['breakdowns']+1)
    def test_07_failed_refresh_retains_last_good_snapshot(self):
        folder=self.folder/'stale';folder.mkdir()
        for name in ['assets.csv','events.csv','observations.csv']:shutil.copyfile(R/'data/demo'/name,folder/name)
        p=folder/'events.csv';e=WorkbookAnalytics(p,allowed_root=self.folder,config_path=self.folder/'stale_config.json');first,_=e.summarize()
        p.write_bytes(b'incomplete CSV save')
        second,_=e.summarize();self.assertTrue(second['source']['stale']);self.assertTrue(second['source']['refresh_error']);self.assertEqual(second['summary'],first['summary']);self.assertEqual(second['source']['sha256'],first['source']['sha256'])
    def test_08_source_change_is_atomic_and_persists(self):
        p=self.bundle/'events.csv';bad=self.folder/'bad.xlsx';bad.write_bytes(b'bad');config=self.folder/'saved_source.json';e=WorkbookAnalytics(p,config_path=config,allowed_root=self.folder)
        e.set_source(str(p));original=config.read_bytes()
        with self.assertRaises(ValueError):e.set_source(str(bad))
        self.assertEqual(config.read_bytes(),original)
        with self.assertRaises(ValueError):e.set_source(str(self.new))
        loaded=WorkbookAnalytics(config_path=config,allowed_root=self.folder);self.assertEqual(loaded.path,p)
    def test_09_routes_export_and_drilldown_are_read_only(self):
        d=self.client.get('/api/analytics?top_n=20');self.assertEqual(d.status_code,200,d.json);self.assertEqual(len(d.json['equipment_rankings']),20)
        e=self.client.get('/api/analytics/events?equipment_id=DEMO-A-005');self.assertEqual(e.status_code,200);self.assertTrue(any(row['linked_work_records']==2 for row in e.json['events']))
        exported=self.client.get('/download/analytics?view=stations');self.assertEqual(exported.status_code,200);self.assertIn('source_sha256',exported.text);self.assertIn('Demo Station A',exported.text)
        self.assertEqual(self.client.post('/api/analytics/refresh',json={}).status_code,403)
        self.assertEqual(self.client.post('/api/analytics/source',json={'source_path':str(self.new)},headers=self.headers).status_code,400)
        with self.app.extensions['maintenance'].connect() as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM maintenance_events').fetchone()[0],0)
    def test_10_csv_escapes_formula_like_text(self):
        data=self.parsed();e=self.app.extensions['analytics'];previous=(e.snapshot,e.meta,e.signature)
        try:
            fixture=self.engine(data);e.snapshot,e.meta,e.signature=fixture.snapshot,fixture.meta,fixture.signature
            e.signature=e._signature(e.path)
            response=self.client.get('/download/analytics?view=events');self.assertEqual(response.status_code,200);self.assertIn("'=HYPERLINK",response.text)
        finally:e.snapshot,e.meta,e.signature=previous

    def test_11_saved_csv_bundle_changes_refresh_counts(self):
        import csv
        folder=self.folder/'csv_reload';folder.mkdir()
        for name in ['assets.csv','events.csv','observations.csv']:shutil.copyfile(R/'data/demo'/name,folder/name)
        e=WorkbookAnalytics(folder/'events.csv',allowed_root=self.folder,config_path=folder/'config.json');first,_=e.summarize()
        p=folder/'events.csv'
        with p.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        original=next(row for row in rows if row['Maintenance Type']!='breakdown');original['Maintenance Type']='breakdown'
        with p.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        second,_=e.summarize();self.assertEqual(second['summary']['breakdowns'],first['summary']['breakdowns']+1)
        self.assertNotEqual(first['source']['sha256'],second['source']['sha256'])

    def test_12_csv_register_and_observations_are_watched(self):
        folder=self.folder/'csv_members';folder.mkdir()
        for name in ['assets.csv','events.csv','observations.csv']:shutil.copyfile(R/'data/demo'/name,folder/name)
        e=WorkbookAnalytics(folder/'events.csv',allowed_root=self.folder,config_path=folder/'config.json');first,_=e.summarize()
        p=folder/'assets.csv';p.write_text(p.read_text(encoding='utf-8').replace('Demo Pump A01','Demo Pump A01 revised'),encoding='utf-8')
        second,_=e.summarize();self.assertNotEqual(first['source']['sha256'],second['source']['sha256'])
        self.assertEqual(next(a['equipment'] for a in second['equipment_totals'] if a['equipment_id']=='DEMO-A-001'),'Demo Pump A01 revised')
        p=folder/'observations.csv';p.write_bytes(p.read_bytes()+b'\n')
        third,_=e.summarize();self.assertNotEqual(second['source']['sha256'],third['source']['sha256'])

if __name__=='__main__':unittest.main(verbosity=2)
