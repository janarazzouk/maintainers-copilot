For the Node.js issue RAG pipeline, I used parent-document retrieval, hybrid dense + keyword retrieval, lightweight technical reranking, and rule-based multi-query rewriting.

The corpus contains GitHub issue language plus exact technical entities such as error codes, stack traces, function names, file paths, and versions. Dense retrieval helps with semantic similarity, while keyword retrieval helps exact code-shaped matches. The reranker boosts candidates that preserve exact technical terms from the user query. Multi-query rewriting is rule-based rather than LLM-based because the dataset structure is known and exact technical terms must not be rewritten incorrectly.

On the 25-example balanced RAG golden set, the final retrieval pipeline achieved Hit@5 = 1.0000, Hit@10 = 1.0000, MRR@10 = 0.9800, and Doc Recall@5 = 1.0000.

