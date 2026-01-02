"""Debug script to test Bell states question."""

from rag_pipeline import query_documents

query = "Bell states are:"
print(f"Query: {query}\n")

result = query_documents(query)

print(f"Answer:\n{result['answer']}")
print(f"\nSources used: {result['chunks_used']}")
