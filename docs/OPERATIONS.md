# Local operation and record backups

## Calendar snapshots

On **Maintenance due**, use **Calendar snapshot** to download an `.ics` file for
enabled tasks with approved dates. The export converts Africa/Lagos due times to
UTC and provides stable task IDs. Meter-only tasks cannot be placed on a calendar
without inventing a date, so they stay in the application's schedule.

Import the file into a calendar application that accepts iCalendar events.
This is a static snapshot, not a subscription or background reminder service.
Re-export after task changes or completions. Recurrence is not encoded as a fixed
calendar rule because this project's repeating interval starts from actual completion.
Import behavior, duplicate handling and calendar notifications depend on the calendar
application. No due dates or database records change during export.

## Back up locally entered records

```sh
python tools/backup_local.py
python tools/backup_local.py --output backups/before-upgrade.zip
```

The tool uses SQLite's online backup API to create a consistent snapshot, verifies
its integrity, and packages the database with a SHA-256 manifest and row counts.
It refuses to overwrite an existing backup. Local databases contain your entered
observations, plans, completions and queue state; keep backups private. `backups/`
is ignored by Git. The backup does not include source CSV/Excel or model artifacts.

For recovery, stop the app, retain a copy of the current database, extract the ZIP
to a separate folder and verify the database's SHA-256 against `manifest.json`.
Use the recovered database with the matching app version only after checking it.
There is deliberately no automatic overwrite or merge of existing records.

## Project layout

`maintenance_app/` remains the application and model package, `tools/` holds
generation/training/backup commands, `notebooks/` contains the walkthrough, and
`docs/` holds data, model, operations and evaluation references. The independent
synthetic CSV demo and the excluded Excel-workbook policy are unchanged.
