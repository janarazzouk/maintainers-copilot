For the Node.js issue RAG pipeline, I used parent-document retrieval, hybrid dense + keyword retrieval, lightweight technical reranking, and rule-based multi-query rewriting.

The corpus contains GitHub issue language plus exact technical entities such as error codes, stack traces, function names, file paths, and versions. Dense retrieval helps with semantic similarity, while keyword retrieval helps exact code-shaped matches. The reranker boosts candidates that preserve exact technical terms from the user query. Multi-query rewriting is rule-based rather than LLM-based because the dataset structure is known and exact technical terms must not be rewritten incorrectly.

On the 25-example balanced RAG golden set, the final retrieval pipeline achieved Hit@5 = 1.0000, Hit@10 = 1.0000, MRR@10 = 0.9800, and Doc Recall@5 = 1.0000.

For RAG retrieval, I evaluated three modes on a balanced 25-question golden set covering bug, docs, feature, and question issues.

Dense-only retrieval already performed strongly, reaching Hit@5 = 1.0000 and MRR@10 = 0.9733. Hybrid dense + keyword retrieval preserved the same score. The final pipeline, which adds rule-based multi-query rewriting and lightweight technical reranking, preserved Hit@5 = 1.0000 and improved MRR@10 to 0.9800.

I kept the final pipeline because the Node.js issue corpus contains exact technical entities such as error codes, function names, versions, stack traces, and file paths. Dense retrieval handles semantic similarity, keyword retrieval protects exact matches, the reranker boosts exact technical matches, and rule-based multi-query rewriting expands the query using dataset-specific GitHub issue language while preserving code terms.