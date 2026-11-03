"""Local Flask interface for maintenance schedules, Gradient Boosting forecasts and workbook analytics."""
from services import MaintenanceService,MODEL,ROOT
from planning import ACTIVE_POLICY,STRATEGY
from analytics import WorkbookAnalytics
from flask import Flask,render_template,jsonify,request,send_file
from werkzeug.exceptions import HTTPException
import io
import csv
import json
import secrets
import zipfile
import pandas as pd

def csv_response(rows,filename):
    frame=pd.DataFrame(rows)
    # Quote text that spreadsheet applications could interpret as formulas.
    for c in frame.columns:
        if frame[c].dtype=='object':frame[c]=frame[c].map(lambda x:"'"+x if isinstance(x,str) and x.lstrip().startswith(('=','+','-','@')) else x)
    return send_file(io.BytesIO(frame.to_csv(index=False).encode('utf-8-sig')),mimetype='text/csv',as_attachment=True,download_name=filename)

def create_app(db_path=None,analytics_source=None,analytics_config=None,analytics_root=None):
    app=Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=12*1024*1024,MAX_FORM_MEMORY_SIZE=64*1024,SECRET_KEY=secrets.token_hex(32))
    service=MaintenanceService(db_path)
    app.extensions['maintenance']=service
    analytics=WorkbookAnalytics(analytics_source,analytics_config,analytics_root,service.manifest.get('source_sha256'))
    app.extensions['analytics']=analytics
    token=secrets.token_urlsafe(32)
    @app.before_request
    def local_access():
        if request.host.split(':')[0] not in ['127.0.0.1','localhost']:return jsonify(error='This app accepts local requests only.'),403
        if request.method=='POST' and not secrets.compare_digest(request.headers.get('X-App-Token',''),token):return jsonify(error='Reload this app before submitting data.'),403
    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff';response.headers['X-Frame-Options']='DENY'
        response.headers['Referrer-Policy']='same-origin';response.headers['Cache-Control']='no-store'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        return response
    @app.errorhandler(ValueError)
    def bad_input(error):return jsonify(error=str(error)),400
    @app.errorhandler(HTTPException)
    def http_error(error):return jsonify(error=error.description),error.code
    @app.route('/')
    def home():return render_template('index.html',app_token=token,analytics_source_path=str(analytics.path))
    @app.route('/api/analytics')
    def analytics_view():return jsonify(analytics.summarize(request.args)[0])
    @app.route('/api/analytics/refresh',methods=['POST'])
    def analytics_refresh():
        payload=request.get_json(silent=True) or {}
        if not isinstance(payload,dict):raise ValueError('Supply analytics filters.')
        return jsonify(analytics.summarize(payload,force=True)[0])
    @app.route('/api/analytics/source',methods=['POST'])
    def analytics_source_view():
        payload=request.get_json(silent=True)
        if not isinstance(payload,dict):raise ValueError('Supply a workbook source path.')
        return jsonify(source=analytics.set_source(payload.get('source_path')))
    @app.route('/api/analytics/events')
    def analytics_events():
        data,events=analytics.summarize(request.args);eid=request.args.get('equipment_id','')
        asset=next((a for a in data['equipment_totals'] if a['equipment_id']==eid),None)
        if not asset:raise ValueError('Select equipment in the current analytics scope.')
        selected=sorted([e for e in events if e['equipment_id']==eid],key=lambda e:e['started_at'] or '',reverse=True)
        return jsonify(equipment=asset['equipment'],equipment_id=eid,total=len(selected),events=selected[:200],source=data['source'])
    @app.route('/api/overview')
    def overview():return jsonify(service.overview())
    @app.route('/api/health')
    def health():return jsonify(application='maintain',ready=True,synthetic_model=True,forecast_version=service.forecasts.manifest.get('version',2) if service.forecasts else 1,algorithm=service.manifest.get('algorithm','CatBoostClassifier'),active_policy=ACTIVE_POLICY,dataset_revision=service.manifest.get('dataset_revision'),source_sha256=service.manifest.get('source_sha256'))
    @app.route('/api/forecast/<equipment_id>')
    def forecast(equipment_id):
        return jsonify(service.equipment_forecast(equipment_id,request.args.get('policy',ACTIVE_POLICY),int(request.args.get('horizon','7'))))
    @app.route('/api/maintenance-status')
    def maintenance_status():return jsonify(service.maintenance_status(request.args.get('equipment_id')))
    @app.route('/api/plans',methods=['GET','POST'])
    def plans():
        if request.method=='GET':return jsonify(plans=service.plans())
        payload=request.get_json(silent=True)
        if not isinstance(payload,dict):raise ValueError('Supply a maintenance plan.')
        return jsonify(plan=service.save_plan(payload)),201
    @app.route('/api/plans/<plan_id>/complete',methods=['POST'])
    def complete_plan(plan_id):
        payload=request.get_json(silent=True)
        if not isinstance(payload,dict):raise ValueError('Supply the completion details.')
        return jsonify(completion=service.complete_plan(plan_id,payload)),201
    @app.route('/api/equipment/<equipment_id>')
    def equipment(equipment_id):return jsonify(service.asset(equipment_id))
    @app.route('/api/predict',methods=['POST'])
    def predict():
        payload=request.get_json(silent=True)
        if not isinstance(payload,dict) or not isinstance(payload.get('observation'),dict):raise ValueError('Supply one observation.')
        mode=payload.get('mode','new');save=payload.get('save',False)
        if not isinstance(save,bool):raise ValueError('Save must be true or false.')
        rows=service.score(pd.DataFrame([payload['observation']]),mode,save,payload.get('policy',ACTIVE_POLICY),int(payload.get('horizon',7)))
        return jsonify(results=rows,saved=save)
    @app.route('/api/predict-batch',methods=['POST'])
    def batch():
        upload=request.files.get('file')
        if not upload:raise ValueError('Choose a CSV file.')
        if not (upload.filename or '').lower().endswith('.csv'):raise ValueError('Upload a CSV file using the downloadable template.')
        try:frame=pd.read_csv(upload.stream)
        except (ValueError,UnicodeError,pd.errors.ParserError) as error:raise ValueError('The CSV could not be read. Use UTF-8 and the template column names.') from error
        return jsonify(results=service.score(frame,request.form.get('mode','new'),request.form.get('save')=='true',request.form.get('policy',ACTIVE_POLICY),int(request.form.get('horizon','7'))))
    @app.route('/api/events',methods=['GET','POST'])
    def events():
        if request.method=='GET':return jsonify(events=service.events())
        payload=request.get_json(silent=True)
        if not isinstance(payload,dict):raise ValueError('Supply an event record.')
        return jsonify(event=service.add_event(payload)),201
    @app.route('/api/queue',methods=['POST'])
    def queue():
        payload=request.get_json(silent=True)
        if not isinstance(payload,dict):raise ValueError('Supply a review status.')
        return jsonify(review=service.queue(payload))
    @app.route('/download/<kind>')
    def download(kind):
        if kind=='analytics':
            data,events=analytics.summarize(request.args);view=request.args.get('view','stations')
            if view=='stations':rows=data['stations']
            elif view=='equipment':rows=data['equipment_totals']
            elif view=='monthly':rows=[{k:v for k,v in row.items() if k!='stations'} for row in data['monthly']]
            elif view=='station-monthly':rows=[{'month':m['month'],'station':station,'breakdowns':count} for m in data['monthly'] for station,count in m['stations'].items()]
            elif view=='events':rows=events
            else:raise ValueError('Choose a valid analytics export view.')
            rows=[{**row,'source_workbook':data['source']['filename'],'source_sha256':data['source']['sha256'],'source_loaded_at':data['source']['loaded_at'],'source_snapshot_stale':data['source']['stale']} for row in rows]
            return csv_response(rows,'breakdown_analytics_'+view+'.csv')
        if kind in ['training-workbook','workbook','training-data'] and service.forecasts:
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
                for name in ['assets.csv','events.csv','observations.csv','provenance.json']:
                    archive.write(ROOT/'data/demo'/name,name)
            buffer.seek(0)
            return send_file(buffer,mimetype='application/zip',as_attachment=True,download_name='synthetic_demo_csv_data.zip')
        if kind=='optimization-report' and service.forecasts and (service.forecasts.directory/'Optimization_report.html').exists():
            return send_file(service.forecasts.directory/'Optimization_report.html',as_attachment=True,download_name='Gradient_Boosting_Optimization_report.html')
        if kind=='report' and service.forecasts:
            return send_file(service.forecasts.directory/'Maintenance_alerts_report.html',as_attachment=True,download_name='Maintenance_alerts_report.html')
        if kind=='policy-comparison' and service.forecasts:
            return csv_response([{'horizon_days':int(h),'strategy':STRATEGY['label'],**spec['policies'][ACTIVE_POLICY]['test_metrics']} for h,spec in service.forecasts.manifest['horizons'].items()],'active_strategy_evaluation.csv')
        files={'template':('new_observations_template.csv','new_observations_template.csv'),
               'example':('historical_example_input.csv','historical_example_input.csv')}
        if kind in files:return send_file(MODEL/files[kind][0],as_attachment=True,download_name=files[kind][1])
        if kind=='plans':return csv_response(service.maintenance_status()['plans'],'maintenance_schedule.csv')
        if kind=='events':return csv_response(service.events(),'observed_maintenance_events.csv')
        if kind=='observations':return csv_response(service.observations(),'supplied_operating_observations.csv')
        if kind=='predictions':return csv_response(service.overview()['equipment'],'historical_equipment_scores.csv')
        return jsonify(error='Unknown download.'),404
    return app

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8060);args=parser.parse_args()
    app=create_app();app.run(host='127.0.0.1',port=args.port,debug=False,use_reloader=False)
