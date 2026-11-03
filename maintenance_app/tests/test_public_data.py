"""Public-release provenance and reproducibility checks."""
from pathlib import Path
import csv,json,subprocess,sys,tempfile,unittest,zipfile
APP=Path(__file__).resolve().parents[1];ROOT=APP.parent
sys.path.insert(0,str(APP/'data/model'))
import maintenance_ml as ml
import pandas as pd
from openpyxl import load_workbook

class PublicDataTests(unittest.TestCase):
    def test_csvs_regenerate_without_external_inputs(self):
        base=APP/'tests/.test_tmp';base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            result=subprocess.run([sys.executable,str(ROOT/'tools/generate_demo.py'),'--output',folder],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            for name in ['assets.csv','observations.csv','events.csv','provenance.json']:
                self.assertEqual((Path(folder)/name).read_bytes(),(APP/'data/demo'/name).read_bytes(),name)

    def test_only_fictional_identifiers_and_sources_are_bundled(self):
        for name in ['assets.csv','observations.csv','events.csv']:
            with (APP/'data/demo'/name).open(encoding='utf-8',newline='') as f:
                for row in csv.DictReader(f):
                    self.assertTrue(row['Equipment ID'].startswith('DEMO-'))
                    station=row.get('Stations',row.get('Station'));self.assertIn(station,[f'Demo Station {c}' for c in 'ABCD'])
                    label=row.get('Record Source',row.get('Event Record Source'))
                    self.assertEqual(label,'Synthetic public demonstration')
        self.assertFalse(list((APP/'data').glob('*.sqlite*')))

    def test_csv_bundle_matches_generated_inputs(self):
        from analytics import WorkbookAnalytics
        e=WorkbookAnalytics(APP/'data/demo/events.csv',model_sha256=json.loads((APP/'data/forecast_models/forecast_manifest.json').read_text())['source_sha256'])
        data,_=e.summarize()
        source=pd.read_csv(APP/'data/demo/events.csv')
        failures=source[source['Maintenance Type']=='breakdown'].drop_duplicates(['Equipment ID','Failure Event ID'])
        self.assertEqual(data['summary']['breakdowns'],len(failures))
        self.assertEqual(data['summary']['maintenance_events'],len(source))
        self.assertEqual(data['source']['source_format'],'csv')
        self.assertTrue(data['source']['matches_model_snapshot'])
        self.assertEqual(data['summary']['registered_equipment'],48)

    def test_horizons_are_purged_and_thresholds_match_validation(self):
        manifest=json.loads((APP/'data/forecast_models/forecast_manifest.json').read_text())
        self.assertTrue(manifest['synthetic_data'])
        self.assertEqual(manifest['dataset_revision'],'Independent synthetic public demonstration')
        self.assertEqual(manifest['algorithm'],'GradientBoostingClassifier')
        for h,spec in manifest['horizons'].items():
            periods=spec['periods']
            for a,b in [('training','calibration'),('calibration','validation'),('validation','test')]:
                self.assertLess(pd.Timestamp(periods[a]['end'])+pd.DateOffset(days=int(h)),pd.Timestamp(periods[b]['start']))
            self.assertGreaterEqual(spec['policies']['catch_more']['validation_metrics']['recall'],.8)
            self.assertFalse(set(spec['features']).intersection(ml.FORBIDDEN))
            self.assertIsNone(spec['transfer_metrics']['catch_more'])

if __name__=='__main__':unittest.main()
