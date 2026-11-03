# Contributing

Use Python 3.12 and the pinned dependencies. Keep changes scoped, describe the user-visible behavior and run `python -m unittest discover -s maintenance_app/tests -v` before opening a pull request.

Keep example data independently generated. Do not commit workplace records, private model weights, credentials, populated SQLite databases, user screenshots or local source paths. New data-generation rules need provenance and consistency checks. Changing a model requires later held-out evaluation, a documented threshold policy and a revised model card; do not tune against published test metrics.

Keep approved schedule rules independent of predicted risk. Add regression coverage for changes to date/hour limits, completion transactions, observation freshness, source refresh and CSV safety. For UI changes check desktop and mobile layouts. State whether AI tools assisted a contribution where relevant.
