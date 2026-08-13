from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, documents, vectorstore):

        self.documents = documents
        self.vectorstore = vectorstore

        tokenized_documents = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)


    # ========================================================
    # KEYWORD SEARCH
    # ========================================================

    def keyword_search(self, query, k=5):

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indexes[:k]:

            results.append(
                self.documents[index]
            )

        return results


    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    def vector_search(self, query, k=5):

        return self.vectorstore.similarity_search(
            query,
            k=k
        )


    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def hybrid_search(self, query, k=10):

        keyword_results = self.keyword_search(
            query,
            k=5
        )

        vector_results = self.vector_search(
            query,
            k=5
        )


        combined = []

        seen = set()


        for document in keyword_results + vector_results:

            text = document.page_content

            if text not in seen:

                combined.append(document)

                seen.add(text)


        return combined[:k]