"""Export approved dated work as a static iCalendar snapshot."""
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import hashlib

def escape(value):
    return str(value or '').replace('\\','\\\\').replace('\r\n','\n').replace('\r','\n').replace('\n','\\n').replace(';','\\;').replace(',','\\,')

def fold(line):
    result=[];part=''
    for character in line:
        if len((part+character).encode('utf-8'))>75:
            result.append(part);part=' '+character
        else: part+=character
    result.append(part)
    return '\r\n'.join(result)

def export_calendar(plans,as_of=None):
    stamp=(as_of or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    lines=['BEGIN:VCALENDAR','VERSION:2.0','PRODID:-//TSobees//Maintenance Schedule//EN',
        'CALSCALE:GREGORIAN','X-WR-CALNAME:Approved maintenance tasks',
        'X-WR-CALDESC:Static snapshot of dated tasks. Meter-only tasks remain in the app.']
    for plan in sorted(plans,key=lambda p:(p.get('due_at') or '',p['id'])):
        if not plan.get('enabled') or not plan.get('due_at'): continue
        date=datetime.fromisoformat(str(plan['due_at']))
        if date.tzinfo is None: date=date.replace(tzinfo=ZoneInfo('Africa/Lagos'))
        due=date.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        uid=hashlib.sha256(str(plan['id']).encode('utf-8')).hexdigest()+'@maintenance.tsobees'
        description='Equipment: '+str(plan['equipment_id'])+'\nStation: '+str(plan.get('station',''))+'\nType: '+str(plan['maintenance_type'])
        if plan.get('due_meter') is not None: description+='\nAlso due at measured meter '+str(plan['due_meter'])+' hours; whichever limit is reached first.'
        description+='\nStatic export. Re-export after completing or editing tasks. Repeats start from actual completion in the app.'
        lines+=['BEGIN:VEVENT','UID:'+uid,'DTSTAMP:'+stamp,'SEQUENCE:'+str(int(plan.get('revision',1))),
            'DTSTART:'+due,'SUMMARY:'+escape(plan['equipment_id']+' · '+plan['task']),
            'DESCRIPTION:'+escape(description),'STATUS:CONFIRMED','END:VEVENT']
    lines+=['END:VCALENDAR','']
    return '\r\n'.join(fold(line) for line in lines).encode('utf-8')
