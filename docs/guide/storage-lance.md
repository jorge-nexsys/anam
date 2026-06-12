# Hybrid Storage & Vector Indexes

AnamDB uses [Lance](https://github.com/lancedb/lance) as its core transactional and analytical storage format, bringing vector search and columnar storage directly to the SQL + Datalog query kernel.

---

## Lance Storage Features

* **Arrow-native:** Native compatibility with Apache Arrow memory formatting, allowing zero-copy data exchanges during execution phases.
* **Vector Search:** Fast Approximate Nearest Neighbor (ANN) index support built directly into table storage formats.
* **Columnar Layout:** Efficient analytical query speeds, ensuring fast projections and aggregates during Datalog evaluations.
* **Versioned History:** Instant snapshots and time-travel capabilities for tracking table history data.

---

## Combining Structured & Vector Data

Because storage is handled natively via Lance, SQL queries can perform vector similarity search alongside standard relational filters and Datalog constraints:

```sql
SELECT id, title, vector_distance(embedding, [0.1, 0.2, ...]) AS dist
FROM articles
WHERE dist < 0.5
  AND status = 'published';
```
