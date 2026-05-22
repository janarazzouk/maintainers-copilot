RAG retrieval was evaluated on 25 hand-curated golden questions. The set is balanced across labels: 7 bug, 6 docs, 6 feature, and 6 question examples.

Final retrieval pipeline:
- Parent-document retrieval
- Dense vector retrieval using BAAI/bge-small-en-v1.5
- PostgreSQL full-text keyword retrieval
- Hybrid dense/keyword scoring
- Lightweight technical reranking
- Rule-based multi-query rewriting

Final metrics:
- Hit@5: 1.0000
- Hit@10: 1.0000
- MRR@10: 0.9800
- Doc Recall@5: 1.0000
- Chunk Recall@10: 0.4200

Chunk recall is lower because the system deduplicates by parent document and returns one best chunk per issue, while the golden set may contain multiple relevant chunks from the same parent issue.