from pathlib import Path
import sys,unittest
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from services import ml
import forecasting as fc
from maintenance_features import add_features
import pandas as pd,numpy as np
class CausalFeatureTests(unittest.TestCase):
    def data(self):
        assets=pd.DataFrame([{ml.ID:'TEST','Stations':'Station A','Equipment Class':'Pump','Simulated Criticality (1-5)':3,'Simulated Typical Daily Hours':16}])
        rows=[]
        for i in range(12):
            r={ml.ID:'TEST',ml.TIME:pd.Timestamp('2026-01-01')+pd.DateOffset(days=i*7),'Station':'Station A','Operating Status':'Running'}
            r.update({c:float(i+1) for c in ml.NUM[:10]});r['Operating Hours Last 24h']=16;r['Inspection Score (0-100)']=i+2;r['Equipment Age (days)']=100+i*7;rows.append(r)
        history=pd.DataFrame(rows)
        events=pd.DataFrame([{ml.ID:'TEST','Maintenance Type':'breakdown','Maintenance Start Timestamp':'2026-01-08 05:00','Maintenance End Timestamp':'2026-01-08 08:00','Operational Downtime (hrs)':3},{ml.ID:'TEST','Maintenance Type':'inspection','Maintenance Start Timestamp':'2026-02-01 00:00','Maintenance End Timestamp':'2026-02-01 00:00','Operational Downtime (hrs)':0}])
        return history,assets,events
    def build(self,h,a,e):
        f,s=fc.build_features(h,a,e);return add_features(f,s,e)
    def test_future_readings_and_events_do_not_change_earlier_features(self):
        h,a,e=self.data();full,sets=self.build(h,a,e);cut=h[ml.TIME].iloc[7]
        short,shortsets=self.build(h[h[ml.TIME]<=cut],a,e[pd.to_datetime(e['Maintenance Start Timestamp'])<=cut])
        pd.testing.assert_frame_equal(short[sets['combined_history']],full.loc[full[ml.TIME]<=cut,sets['combined_history']].reset_index(drop=True))
        poisoned=h.copy();poisoned.loc[poisoned[ml.TIME]>cut,'Bearing Temperature (C)']=10000
        other,_=self.build(poisoned,a,e);pd.testing.assert_frame_equal(full.loc[full[ml.TIME]<=cut,sets['combined_history']],other.loc[other[ml.TIME]<=cut,sets['combined_history']])
    def test_ongoing_failure_is_not_completed_history(self):
        h,a,e=self.data();h.loc[1,ml.TIME]=pd.Timestamp('2026-01-08 06:00');f,s=self.build(h,a,e)
        row=f[f[ml.TIME]==pd.Timestamp('2026-01-08 06:00')].iloc[0]
        self.assertEqual(row['Completed breakdowns 30d'],0);self.assertTrue(pd.isna(row['Days since completed breakdown']))
        self.assertEqual(f.iloc[2]['Completed breakdowns 30d'],1)
    def test_features_exclude_identifiers_and_future_outcomes(self):
        h,a,e=self.data();h[ml.TARGET]=1;h['Next Failure Lead Time (hrs)']=0
        f,s=self.build(h,a,e)
        for names in s.values():self.assertFalse(set(names).intersection(ml.FORBIDDEN))
        numeric=[c for c in s['combined_history'] if c not in ml.CAT];self.assertFalse(np.isinf(f[numeric].to_numpy()).any())
if __name__=='__main__':unittest.main(verbosity=2)
