# Public-edition data handling

The public edition was built as a separate folder from source code, rather than by renaming an exported workplace workbook. Its generator reads no original register or maintenance history. All bundled CSV rows are fictional, and the fitted pipelines were trained afresh from those generated records. Private trained models and historical private reports were excluded.

Release checks reconcile generated CSV inputs with analytics and model context, check generic identifiers and model encoder categories, exclude both Excel files from the public Git index and archive, and scan text for common credential patterns. The walkthrough notebook contains no saved execution outputs. Screenshots were captured from an isolated instance containing only the fictional dataset.

Local databases, paths, logs, caches and runtime dependencies are excluded from version control and the release archive. These checks reduce exposure risk but do not constitute a comprehensive security audit. Later real observations, workbooks and screenshots must stay out of the public repository. Use ignored local_data/ and review staged changes before each push.
