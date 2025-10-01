```bash
$ bash -lc python - <<'PY'
import psycopg
conn = psycopg.connect('postgresql://langchain:langchain@localhost:6024/langchain')
with conn.cursor() as cur:
    cur.execute("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname='langchain_pg_embedding_pkey'")
    print(cur.fetchone())
conn.close()
PY
```