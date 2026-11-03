# Public repository setup

Suggested repository name: `predictive-maintenance-dashboard`.

Suggested description: **A Flask maintenance dashboard with approved service schedules, Gradient Boosting risk forecasts and Excel-linked breakdown analytics. Includes independent synthetic demo data.**

Suggested topics: `predictive-maintenance`, `gradient-boosting`, `flask`, `machine-learning`, `data-visualization`, `synthetic-data`, `python`.

Publish this public-edition repository folder, rather than its parent project or the earlier private package. Commit the MIT license, demo data, models, documentation and source. Keep local databases, real workbooks, credentials and temporary artifacts excluded by `.gitignore`. Review the staged files before pushing.

After uploading, check the Project checks workflow, confirm the repository visibility is public, and review the README screenshots. Enable private vulnerability reporting in repository settings if desired. The workflow configuration is included; successful local tests do not imply a remote workflow has already run.

This publishes source code. A hosted web demo would require separate deployment and access controls; GitHub Pages cannot execute this Flask server. Keep the included application local unless those requirements are implemented.
