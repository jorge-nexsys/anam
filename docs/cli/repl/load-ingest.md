# REPL: loading & Ingesting Data

Learn how to import CSV datasets and load versioned Lance tables into the active catalog.

---

## `.ingest`
Converts a raw CSV data file and writes it into a versioned, Arrow-backed Lance table format.

* **Syntax**: `.ingest <csv_path> [lance_output_path]`
* **Example**:
  ```sql
  anam> .ingest demo/data/transactions.csv demo/data/transactions.lance
  ✓ Ingested 100,000 rows (4.7 MB CSV → Lance columnar)
  ```

---

## `.load`
Registers an existing Lance table into the current query session catalog, making it queryable with SQL.

* **Syntax**: `.load <lance_path> [table_name]`
* **Example**:
  ```sql
  anam> .load demo/data/transactions.lance txns
  Registered table 'txns' from demo/data/transactions.lance
  ```
