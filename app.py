import streamlit as st

from rag_pipeline import (
    load_pdf,
    split_documents,
    create_vector_store,
    generate_answer
)

from memory import (
    initialize_memory,
    add_message,
    get_chat_history,
    clear_memory
)

from hybrid_retriever import HybridRetriever
from reranker import DocumentReranker

from evaluation import (
    evaluation_questions,
    is_relevant,
    answer_correctness,
    calculate_metrics,
    calculate_average_answer_correctness
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Level 3 RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Level 3 PDF RAG Chatbot")

st.write(
    "Conversational Memory + Hybrid Retrieval + Reranking"
)


# ============================================================
# INITIALIZE MEMORY
# ============================================================

initialize_memory()


# ============================================================
# INITIALIZE RERANKER
# ============================================================

if "reranker" not in st.session_state:

    with st.spinner("Loading reranker model..."):

        st.session_state.reranker = DocumentReranker()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 PDF Upload")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        clear_memory()

        st.rerun()


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file:

    with open(
        "uploaded.pdf",
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )


    # Process only once per session
    if "documents" not in st.session_state:

        with st.spinner(
            "Processing PDF..."
        ):

            # ----------------------------------------------
            # LOAD PDF
            # ----------------------------------------------

            documents = load_pdf(
                "uploaded.pdf"
            )


            # ----------------------------------------------
            # SPLIT INTO CHUNKS
            # ----------------------------------------------

            chunks = split_documents(
                documents
            )


            # ----------------------------------------------
            # CREATE VECTOR STORE
            # ----------------------------------------------

            vectorstore = create_vector_store(
                chunks
            )


            # ----------------------------------------------
            # SAVE IN SESSION
            # ----------------------------------------------

            st.session_state.documents = chunks

            st.session_state.vectorstore = vectorstore


            # ----------------------------------------------
            # HYBRID RETRIEVER
            # ----------------------------------------------

            st.session_state.hybrid = HybridRetriever(
                chunks,
                vectorstore
            )


        st.success(
            f"PDF processed successfully! "
            f"{len(chunks)} chunks created."
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in get_chat_history():

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your PDF..."
)


if question:

    # ========================================================
    # CHECK PDF
    # ========================================================

    if "hybrid" not in st.session_state:

        st.error(
            "⚠️ Please upload a PDF first."
        )

        st.stop()


    # ========================================================
    # DISPLAY USER QUESTION
    # ========================================================

    add_message(
        "user",
        question
    )

    with st.chat_message("user"):

        st.write(question)


    # ========================================================
    # TASK 2 — HYBRID RETRIEVAL
    # ========================================================

    with st.spinner(
        "🔍 Hybrid search: Keyword + Vector..."
    ):

        retrieved_documents = (
            st.session_state.hybrid.hybrid_search(
                question,
                k=10
            )
        )


    # ========================================================
    # TASK 3 — RERANKING
    # ========================================================

    with st.spinner(
        "🔄 Reranking top 10 documents..."
    ):

        reranked_documents = (
            st.session_state.reranker.rerank(
                question,
                retrieved_documents,
                top_k=3
            )
        )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    with st.spinner(
        "🤖 Generating answer..."
    ):

        answer = generate_answer(
            question,
            reranked_documents,
            get_chat_history()
        )


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    with st.chat_message("assistant"):

        st.write(answer)


    # ========================================================
    # SAVE ANSWER TO MEMORY
    # ========================================================

    add_message(
        "assistant",
        answer
    )


    # ========================================================
    # RETRIEVAL INFORMATION
    # ========================================================

    with st.expander(
        "🔍 Retrieval Details"
    ):

        st.write(
            f"**Hybrid Retrieval:** "
            f"{len(retrieved_documents)} documents"
        )

        st.write(
            f"**After Reranking:** "
            f"{len(reranked_documents)} documents"
        )


        st.divider()

        st.write(
            "**Top 3 Reranked Documents:**"
        )


        for i, document in enumerate(
            reranked_documents,
            1
        ):

            st.markdown(
                f"### 📄 Document {i}"
            )

            st.write(
                document.page_content
            )


# ============================================================
# TASK 4 — RETRIEVAL EVALUATION
# ============================================================

st.divider()

st.header("📊 Retrieval Evaluation")

st.write(
    "Evaluate the RAG system using 10 predefined questions."
)


if st.button(
    "▶ Run Evaluation",
    use_container_width=True
):

    # --------------------------------------------------------
    # CHECK PDF
    # --------------------------------------------------------

    if "hybrid" not in st.session_state:

        st.error(
            "⚠️ Please upload a PDF before running evaluation."
        )

        st.stop()


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []


    progress = st.progress(0)


    status = st.empty()


    # ========================================================
    # RUN 10 QUESTIONS
    # ========================================================

    for index, item in enumerate(
        evaluation_questions
    ):

        question = item["question"]

        expected_answer = item[
            "expected_answer"
        ]


        status.write(
            f"Evaluating question "
            f"{index + 1}/"
            f"{len(evaluation_questions)}..."
        )


        # ----------------------------------------------------
        # HYBRID RETRIEVAL
        # ----------------------------------------------------

        retrieved_documents = (
            st.session_state.hybrid.hybrid_search(
                question,
                k=10
            )
        )


        # ----------------------------------------------------
        # RERANK
        # ----------------------------------------------------

        reranked_documents = (
            st.session_state.reranker.rerank(
                question,
                retrieved_documents,
                top_k=3
            )
        )


        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        generated_answer = generate_answer(
            question,
            reranked_documents,
            []
        )


        # ----------------------------------------------------
        # RETRIEVAL RELEVANCE
        # ----------------------------------------------------

        retrieval_correct = is_relevant(
            expected_answer,
            retrieved_documents
        )


        # ----------------------------------------------------
        # ANSWER CORRECTNESS
        # ----------------------------------------------------

        correctness = answer_correctness(
            generated_answer,
            expected_answer
        )


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append({

            "question":
                question,

            "expected_answer":
                expected_answer,

            "generated_answer":
                generated_answer,

            "retrieval_correct":
                retrieval_correct,

            "answer_correctness":
                correctness

        })


        # ----------------------------------------------------
        # UPDATE PROGRESS
        # ----------------------------------------------------

        progress.progress(
            (index + 1)
            /
            len(evaluation_questions)
        )


    status.success(
        "✅ Evaluation completed!"
    )


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    metrics = calculate_metrics(
        results
    )


    average_correctness = (
        calculate_average_answer_correctness(
            results
        )
    )


    # ========================================================
    # DISPLAY METRICS
    # ========================================================

    st.subheader(
        "📈 Evaluation Metrics"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Retrieval Accuracy",
            f"{metrics['accuracy']}%"
        )


    with col2:

        st.metric(
            "Precision",
            f"{metrics['precision']}%"
        )


    with col3:

        st.metric(
            "Recall",
            f"{metrics['recall']}%"
        )


    with col4:

        st.metric(
            "Answer Correctness",
            f"{average_correctness}%"
        )


    # ========================================================
    # DETAILED RESULTS
    # ========================================================

    st.subheader(
        "📝 Detailed Results"
    )


    for i, result in enumerate(
        results,
        1
    ):

        with st.expander(
            f"Question {i}: "
            f"{result['question']}"
        ):

            st.write(
                "**Expected Answer:**"
            )

            st.write(
                result["expected_answer"]
            )


            st.write(
                "**Generated Answer:**"
            )

            st.write(
                result["generated_answer"]
            )


            st.write(
                "**Retrieval Result:**"
            )


            if result["retrieval_correct"]:

                st.success(
                    "✅ Relevant information retrieved"
                )

            else:

                st.error(
                    "❌ Relevant information not retrieved"
                )


            st.write(
                "**Answer Correctness:** "
                f"{result['answer_correctness'] * 100:.2f}%"
            )