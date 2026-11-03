"""Train fixed-config public demo models, with horizon-purged chronological splits."""
from pathlib import Path
import sys,json,hashlib,argparse,html
ROOT=Path(__file__).resolve().parents[1];APP=ROOT/'maintenance_app'
sys.path.insert(0,str(APP/'data/model'));sys.path.insert(0,str(APP))
import maintenance_ml as ml
import forecasting as fc
import gradient_boosting as gb
import pandas as pd,numpy as np,joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,brier_score_loss

def measures(y,p,threshold):
    y=np.asarray(y,dtype=int);p=np.asarray(p);alert=p>=threshold
    tp=int(((y==1)&alert).sum());fp=int(((y==0)&alert).sum());fn=int(((y==1)&~alert).sum());tn=int(((y==0)&~alert).sum())
    return {'n':len(y),'failures':int(y.sum()),'true_positive':tp,'false_positive':fp,'false_negative':fn,'true_negative':tn,
        'recall':tp/(tp+fn) if tp+fn else 0.,'precision':tp/(tp+fp) if tp+fp else 0.,
        'alert_fraction':float(alert.mean()),'prevalence':float(y.mean()),'accuracy':float(accuracy_score(y,alert)),
        'balanced_accuracy':float(balanced_accuracy_score(y,alert)),
        'average_precision':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)) if len(set(y))>1 else None,
        'brier_score':float(brier_score_loss(y,p)),'inspections_per_detected_failure':(tp+fp)/tp if tp else None}

def train(data,out):
    out.mkdir(parents=True,exist_ok=True);model_dir=APP/'data/model'
    assets=pd.read_csv(data/'assets.csv');history=pd.read_csv(data/'observations.csv');events=pd.read_csv(data/'events.csv')
    history[ml.TIME]=pd.to_datetime(history[ml.TIME]);history=history.sort_values([ml.ID,ml.TIME]).reset_index(drop=True)
    for c in ['Maintenance Start Timestamp','Maintenance End Timestamp']:events[c]=pd.to_datetime(events[c])
    # Labels are constructed from distinct episode start times, never follow-up work dates.
    failures=events[events['Maintenance Type']=='breakdown'].groupby([ml.ID,'Failure Event ID'])['Maintenance Start Timestamp'].min().reset_index()
    ends=pd.Timestamp(json.loads((data/'provenance.json').read_text())['coverage_end'])
    frame,sets=fc.build_features(history,assets,events);names=sets['core']
    dates=sorted(frame[ml.TIME].unique());cal_start=pd.Timestamp(dates[25]);val_start=pd.Timestamp(dates[34]);test_start=pd.Timestamp(dates[43])
    config={'feature_set':'core','weight_mode':'none','max_depth':2,'learning_rate':.05,'min_samples_leaf':25,'subsample':.8}
    manifest={'version':1,'algorithm':'GradientBoostingClassifier','model_display_name':'Gradient Boosting (synthetic demo)',
        'default_policy':'catch_more','policy_definitions':{'catch_more':{'label':'Catch more failures','definition':'Highest validation threshold reaching 80% failure-window recall.'}},
        'synthetic_data':True,'dataset_revision':'Independent synthetic public demonstration','source_workbook':'synthetic_maintenance_demo.xlsx',
        'coverage_start':str(dates[0]),'coverage_end':str(ends),'context_last_timestamp':str(history[ml.TIME].max()),'seed':271828,
        'evaluation_limit':'Held-out dates within an invented simulation. No real records or field validation; no equipment-transfer evaluation.',
        'search_trials':1,'cross_validation_folds':0,'reserved_equipment_ids':[],
        'dataset_summary':{'assets':len(assets),'observations':len(history),'source':'Synthetic public demonstration'},'horizons':{}}
    digest=ml.demo_source_digest(data);manifest['source_sha256']=digest
    manifest['source_workbook']='Synthetic CSV demo bundle';manifest['source_format']='csv'
    manifest['source_files']=['assets.csv','events.csv','observations.csv']
    raw=[];masks_by_h={};labels_by_h={};models={}
    for h in fc.HORIZONS:
        label=np.full(len(frame),np.nan)
        for eid,rows in frame.groupby(ml.ID):
            t=rows[ml.TIME].to_numpy(dtype='datetime64[ns]');f=failures[failures[ml.ID]==eid]['Maintenance Start Timestamp'].to_numpy(dtype='datetime64[ns]')
            label[rows.index]=((f[None,:]>t[:,None])&(f[None,:]<=t[:,None]+np.timedelta64(h,'D'))).any(axis=1)
        finish=frame[ml.TIME]+pd.DateOffset(days=h)
        eligible=(finish<=ends)&frame['Operating Status'].isin(['Running','Degraded'])
        label[~eligible]=np.nan;labels_by_h[h]=label
        masks={'training':eligible&(finish<cal_start),'calibration':eligible&(frame[ml.TIME]>=cal_start)&(finish<val_start),
               'validation':eligible&(frame[ml.TIME]>=val_start)&(finish<test_start),'test':eligible&(frame[ml.TIME]>=test_start)}
        masks_by_h[h]=masks
        for name,mask in masks.items():
            assert mask.sum()>0 and len(np.unique(label[mask]))==2,(h,name)
        model=gb.make_pipeline(names,config,271828+h,80)
        gb.fit(model,frame.loc[masks['training'],names],label[masks['training']].astype(int),'none')
        score=model.decision_function(frame[names]);cal_mask=masks['calibration']
        mapping=LogisticRegression(C=1.,random_state=271828).fit(score[cal_mask,None],label[cal_mask].astype(int))
        slope=float(mapping.coef_[0,0]);calibration={'method':'sigmoid','slope':slope,'intercept':float(mapping.intercept_[0])} if slope>0 else {'method':'identity'}
        probability=ml.calibration_predict(score,calibration);raw.append(probability);models[h]=model
        filename=f'demo_{h}d.joblib';joblib.dump(model,out/filename,compress=3)
        spec={'model_files':[filename],'trees_per_model':80,'features':names,'configuration':config,'calibration':calibration,
              'periods':{name:{'start':str(frame.loc[m,ml.TIME].min()),'end':str(frame.loc[m,ml.TIME].max()),'n':int(m.sum())} for name,m in masks.items()},
              'policies':{},'transfer_metrics':{'catch_more':None}}
        manifest['horizons'][str(h)]=spec
    projected=fc.coherent_probabilities(np.column_stack(raw));report_rows=[]
    for j,h in enumerate(fc.HORIZONS):
        spec=manifest['horizons'][str(h)];masks=masks_by_h[h];label=labels_by_h[h];v=masks['validation'];p=projected[:,j]
        thresholds=np.unique(p[v]);valid=[t for t in thresholds if ((p[v]>=t)&(label[v]==1)).sum()/int((label[v]==1).sum())>=.8]
        threshold=float(max(valid));test=masks['test']
        spec['policies']['catch_more']={'threshold':threshold,'validation_metrics':measures(label[v],p[v],threshold),'test_metrics':measures(label[test],p[test],threshold)}
        report_rows.append({'horizon_days':h,**spec['policies']['catch_more']['test_metrics']})
        check=joblib.load(out/spec['model_files'][0]);np.testing.assert_allclose(check.decision_function(frame[names].tail(10)),models[h].decision_function(frame[names].tail(10)))
        saved=pd.DataFrame({'equipment_id':frame.loc[test,ml.ID],'observed_at':frame.loc[test,ml.TIME], 'target':label[test].astype(int),'probability':p[test]})
        saved.to_csv(out/f'test_predictions_{h}d.csv',index=False)
    ml.save_json(out/'forecast_manifest.json',manifest)
    # Assets, inference context and source artifacts contain generated rows only.
    snapshot=history.sort_values([ml.ID,ml.TIME]).groupby(ml.ID).tail(1).copy()
    indexes=frame.groupby(ml.ID).tail(1).index
    snapshot=snapshot.set_index(ml.ID).loc[frame.loc[indexes,ml.ID]].reset_index()
    snapshot['Seven-day failure probability']=np.where(snapshot['Operating Status'].isin(['Running','Degraded']),projected[indexes,0],np.nan)
    snapshot['Alert threshold']=manifest['horizons']['7']['policies']['catch_more']['threshold']
    for directory in [out,model_dir]:
        assets.to_csv(directory/'asset_metadata.csv',index=False)
        history.to_csv(directory/'historical_context.csv',index=False)
        example=history.groupby(ml.ID).tail(1)
        example.to_csv(directory/'historical_example_input.csv',index=False)
        example.iloc[:0].to_csv(directory/'new_observations_template.csv',index=False)
    snapshot.to_csv(out/'latest_snapshot_balanced.csv',index=False)
    snapshot.to_csv(model_dir/'latest_snapshot_predictions.csv',index=False)
    events.to_csv(out/'maintenance_events.csv',index=False)
    ml.save_json(model_dir/'model_manifest.json',{'features':names,'source_sha256':digest})
    ml.save_json(model_dir/'evaluation.json',{'reserved_equipment_count':0,'cross_validation_folds':0})
    metrics=manifest['horizons']['7']['policies']['catch_more']['test_metrics']
    ml.save_json(out/'evaluation.json',{'source':'Synthetic public demonstration','horizons':report_rows})
    lines=['# Synthetic demonstration evaluation','','All results are generated from fictional data. This is not a report of field accuracy.','','| Horizon | Test windows | Positive windows | Recall | Precision | Flagged | ROC-AUC | AP |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in report_rows:lines.append(f"| {r['horizon_days']} days | {r['n']} | {r['failures']} | {r['recall']:.1%} | {r['precision']:.1%} | {r['alert_fraction']:.1%} | {r['roc_auc']:.3f} | {r['average_precision']:.3f} |")
    lines+=['','One fixed configuration per horizon; no test-based model search. Thresholds are set on separate validation dates, targeting at least 80% recall there. Test recall can differ. Calibration uses an earlier separate period. Outcome windows are purged at boundaries.','', 'These windows overlap and are not counts of unique physical failures. The simulation itself defines condition/failure associations. No unseen-equipment transfer result, real alert workload reduction or field validation is claimed. See MODEL_CARD.md and the machine-readable evaluation under maintenance_app/data/forecast_models/.']
    (ROOT/'docs/EVALUATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    ml.save_json(APP/'data/active_strategy.json',{'active_policy':'catch_more','synthetic_demo':True})
    columns=['horizon_days','n','failures','recall','precision','alert_fraction','roc_auc','average_precision']
    table=pd.DataFrame(report_rows)[columns].to_html(index=False,float_format=lambda x:f'{x:.3f}')
    report='<!doctype html><html lang="en"><meta charset="utf-8"><title>Synthetic demonstration evaluation</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:20px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>Synthetic demonstration evaluation</h1><p>All equipment, readings and events are fictional. Fixed configuration; fit on earlier dates, sigmoid mapping on a later calibration period, thresholds on validation, then evaluated on held-out dates. Outcome horizons are purged across boundaries.</p>'+table+'<p>Metrics count overlapping equipment-observation windows, not necessarily distinct failures. No transfer or field validation was performed. Approved schedules remain independent of the classifier.</p></html>'
    (out/'Maintenance_alerts_report.html').write_text(report,encoding='utf-8')
    ml.save_json(out/'training_checks.json',{'result':'PASS','source':'Generated public demonstration only','source_sha256':digest,
        'chronological_horizon_purge':True,'fixed_configuration':config,'trees':80,'serialization_roundtrip':True,
        'separate_calibration_and_threshold_validation':True,'coherent_horizons':True,'test_not_used_for_selection':True})
    print(json.dumps({'result':'PASS','public_demo':True,'assets':len(assets),'observations':len(history),'evaluation':report_rows},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=APP/'data/demo');p.add_argument('--output',type=Path,default=APP/'data/forecast_models');a=p.parse_args();train(a.data,a.output)
