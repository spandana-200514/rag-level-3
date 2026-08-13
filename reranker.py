from sentence_transformers import CrossEncoder


class DocumentReranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )


    def rerank(
        self,
        query,
        documents,
        top_k=3
    ):

        if not documents:

            return []


        pairs = [
            (
                query,
                document.page_content
            )
            for document in documents
        ]


        scores = self.model.predict(pairs)


        scored_documents = list(
            zip(documents, scores)
        )


        scored_documents.sort(
            key=lambda x: x[1],
            reverse=True
        )


        return [
            document
            for document, score
            in scored_documents[:top_k]
        ]