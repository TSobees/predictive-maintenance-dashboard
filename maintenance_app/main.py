"""Launch the local app and open its interface in the default browser."""
import threading
import urllib.request
import webbrowser
import time
import json
from app import create_app

def open_when_ready(url):
    for _ in range(15):
        try:
            with urllib.request.urlopen(url,timeout=2):pass
            webbrowser.open(url);return
        except OSError:time.sleep(1)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8060);parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    url=f'http://127.0.0.1:{args.port}'
    try:
        with urllib.request.urlopen(url+'/api/health',timeout=2) as response:
            already_running=json.load(response).get('application')=='maintain'
    except (OSError,ValueError):already_running=False
    if already_running:
        print(f'Maintain is already running at {url}',flush=True)
        if not args.no_browser:webbrowser.open(url)
        raise SystemExit(0)
    app=create_app()
    if not args.no_browser:threading.Thread(target=open_when_ready,args=(url,),daemon=True).start()
    print(f'Maintain dashboard: {url}',flush=True)
    app.run(host='127.0.0.1',port=args.port,debug=False,use_reloader=False)
