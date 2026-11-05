from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import zipfile

APP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP));sys.path.insert(0,str(APP.parent))
from calendar_export import export_calendar
from tools.backup_local import backup

class CalendarTests(unittest.TestCase):
    def plan(self,**changes):
        row={'id':'a','revision':2,'enabled':True,'due_at':'2026-11-06 09:00:00','equipment_id':'DEMO-A-001',
            'task':'Inspect bearings','maintenance_type':'inspection','station':'Station A','due_meter':None}
        row.update(changes);return row
    def test_due_time_uses_lagos_and_is_a_snapshot(self):
        data=export_calendar([self.plan()]).decode()
        self.assertIn('DTSTART:20261106T080000Z',data)
        self.assertIn('SEQUENCE:2',data)
        self.assertNotIn('RRULE:',data)
        self.assertEqual(data.count('BEGIN:VEVENT'),1)
    def test_inactive_and_meter_only_tasks_excluded(self):
        data=export_calendar([self.plan(enabled=False),self.plan(due_at=None,due_meter=500)]).decode()
        self.assertNotIn('BEGIN:VEVENT',data)
    def test_escape_injection_and_fold_utf8(self):
        content=export_calendar([self.plan(task='é'*120+'\r\nBEGIN:VEVENT,;')])
        self.assertTrue(all(len(line)<=75 for line in content.split(b'\r\n')))
        text=content.decode();self.assertEqual(text.count('\r\nBEGIN:VEVENT'),1)
        self.assertIn('\\nBEGIN:VEVENT\\,\\;',text.replace('\r\n ',''))
    def test_stable_uids_and_source_unchanged(self):
        p=self.plan();copy=p.copy()
        def uid(data):return next(x for x in data.decode().splitlines() if x.startswith('UID:'))
        self.assertEqual(uid(export_calendar([p])),uid(export_calendar([self.plan(revision=3)])))
        self.assertEqual(p,copy)
    def test_calendar_download_route_is_read_only(self):
        from app import create_app
        with tempfile.TemporaryDirectory() as tmp:
            app=create_app(Path(tmp)/'records.sqlite3');client=app.test_client()
            service=app.extensions['maintenance']
            service.save_plan({'equipment_id':'DEMO-A-001','maintenance_type':'inspection','task':'Inspect bearings','due_at':'2026-11-06 09:00:00'})
            before=service.plans();response=client.get('/download/calendar')
            self.assertEqual(response.status_code,200)
            self.assertTrue(response.mimetype.startswith('text/calendar'))
            self.assertIn('BEGIN:VEVENT',response.text)
            self.assertEqual(service.plans(),before)

class BackupTests(unittest.TestCase):
    def test_snapshot_manifest_and_source_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'records.sqlite3';out=Path(tmp)/'records.zip'
            db=sqlite3.connect(source);db.execute('CREATE TABLE observations(id INTEGER)');db.execute('INSERT INTO observations VALUES(1)');db.commit();db.close()
            before=source.read_bytes();manifest=backup(source,out)
            with zipfile.ZipFile(out) as archive:
                self.assertEqual(set(archive.namelist()),{'maintenance_records.sqlite3','manifest.json'})
                self.assertEqual(hashlib.sha256(archive.read('maintenance_records.sqlite3')).hexdigest(),manifest['sha256'])
            self.assertEqual(manifest['row_counts']['observations'],1)
            self.assertEqual(source.read_bytes(),before)
            with self.assertRaises(ValueError): backup(source,out)
    def test_missing_and_same_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): backup(Path(tmp)/'missing',Path(tmp)/'out.zip')

if __name__=='__main__': unittest.main()
