"""Create a consistent local SQLite snapshot without opening the app or changing records."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def backup(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if not source.is_file(): raise ValueError('No local record database exists at '+str(source))
    if source==output: raise ValueError('Backup output must differ from the source database.')
    if output.exists(): raise ValueError('Choose a new backup filename; existing files are not overwritten.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        snapshot=Path(temporary)/'maintenance_records.sqlite3'
        src=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)
        dst=sqlite3.connect(snapshot)
        try:
            src.backup(dst)
            if dst.execute('PRAGMA integrity_check').fetchone()[0]!='ok': raise ValueError('SQLite integrity check failed.')
            tables={row[0] for row in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            counts={name:dst.execute('SELECT COUNT(*) FROM '+name).fetchone()[0] for name in
                ['maintenance_plans','maintenance_completions','observations','maintenance_events','inspection_queue'] if name in tables}
        finally: dst.close();src.close()
        digest=hashlib.sha256(snapshot.read_bytes()).hexdigest()
        manifest={'format':1,'created_at':datetime.now(timezone.utc).isoformat(),'database':'maintenance_records.sqlite3',
            'sha256':digest,'row_counts':counts,'contents':'Local records only. Source CSV/Excel and trained models are not included.'}
        with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as archive:
            archive.write(snapshot,'maintenance_records.sqlite3')
            archive.writestr('manifest.json',json.dumps(manifest,indent=2))
    return manifest

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,default=ROOT/'maintenance_app/data/maintenance_records.sqlite3')
    p.add_argument('--output',type=Path,default=ROOT/'backups'/('records-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.zip'))
    a=p.parse_args(argv)
    try: result=backup(a.database,a.output)
    except (OSError,ValueError,sqlite3.Error) as error: p.error(str(error))
    print(json.dumps(result,indent=2));print('Backup saved locally: '+str(a.output.resolve()))

if __name__=='__main__': main()
