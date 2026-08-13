import re


# ============================================================
# 10 EVALUATION QUESTIONS
# ============================================================

evaluation_questions = [

    {
        "question": "What is machine learning?",
        "expected_answer":
            "Machine learning is a method where computers learn patterns from data."
    },

    {
        "question": "What are the types of machine learning?",
        "expected_answer":
            "The main types are supervised, unsupervised and reinforcement learning."
    },

    {
        "question": "What is supervised learning?",
        "expected_answer":
            "Supervised learning uses labelled training data."
    },

    {
        "question": "What is unsupervised learning?",
        "expected_answer":
            "Unsupervised learning finds patterns in unlabelled data."
    },

    {
        "question": "What is reinforcement learning?",
        "expected_answer":
            "Reinforcement learning learns using rewards and penalties."
    },

    {
        "question": "What is deep learning?",
        "expected_answer":
            "Deep learning uses neural networks with multiple layers."
    },

    {
        "question": "What is a neural network?",
        "expected_answer":
            "A neural network is a computational model inspired by biological neurons."
    },

    {
        "question": "What is classification?",
        "expected_answer":
            "Classification predicts a category or class."
    },

    {
        "question": "What is regression?",
        "expected_answer":
            "Regression predicts a continuous numerical value."
    },

    {
        "question": "What is overfitting?",
        "expected_answer":
            "Overfitting occurs when a model performs well on training data but poorly on unseen data."
    }
]


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    return text


# ============================================================
# ANSWER CORRECTNESS
# ============================================================

def answer_correctness(
    generated_answer,
    expected_answer
):

    generated_words = set(
        normalize_text(
            generated_answer
        ).split()
    )

    expected_words = set(
        normalize_text(
            expected_answer
        ).split()
    )

    if not expected_words:
        return 0.0

    common_words = (
        generated_words &
        expected_words
    )

    score = (
        len(common_words)
        /
        len(expected_words)
    )

    return round(score, 2)


# ============================================================
# RETRIEVAL RELEVANCE
# ============================================================

def is_relevant(
    expected_answer,
    retrieved_documents
):

    expected_words = set(
        normalize_text(
            expected_answer
        ).split()
    )

    if not expected_words:
        return False

    retrieved_text = " ".join(
        document.page_content
        for document in retrieved_documents
    )

    retrieved_words = set(
        normalize_text(
            retrieved_text
        ).split()
    )

    common_words = (
        expected_words &
        retrieved_words
    )

    relevance_score = (
        len(common_words)
        /
        len(expected_words)
    )

    return relevance_score >= 0.30


# ============================================================
# PRECISION
# ============================================================

def calculate_precision(
    relevant_retrieved,
    total_retrieved
):

    if total_retrieved == 0:
        return 0.0

    return round(
        relevant_retrieved
        /
        total_retrieved,
        2
    )


# ============================================================
# RECALL
# ============================================================

def calculate_recall(
    relevant_retrieved,
    total_relevant
):

    if total_relevant == 0:
        return 0.0

    return round(
        relevant_retrieved
        /
        total_relevant,
        2
    )


# ============================================================
# RETRIEVAL ACCURACY
# ============================================================

def calculate_accuracy(
    results
):

    if not results:
        return 0.0

    correct = sum(
        1
        for result in results
        if result["retrieval_correct"]
    )

    return round(
        correct / len(results),
        2
    )


# ============================================================
# ALL RETRIEVAL METRICS
# ============================================================

def calculate_metrics(results):

    if not results:

        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }


    # Number of questions where
    # relevant information was retrieved

    relevant_retrieved = sum(
        1
        for result in results
        if result["retrieval_correct"]
    )


    # Total evaluation questions

    total_questions = len(results)


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = calculate_accuracy(
        results
    )


    # --------------------------------------------------------
    # Precision
    #
    # Relevant retrieved / Total retrieved
    #
    # For this question-level evaluation,
    # each question represents one retrieval task.
    # --------------------------------------------------------

    precision = calculate_precision(
        relevant_retrieved,
        total_questions
    )


    # --------------------------------------------------------
    # Recall
    #
    # Relevant retrieved / Total relevant
    #
    # We have one expected relevant result
    # for each evaluation question.
    # --------------------------------------------------------

    recall = calculate_recall(
        relevant_retrieved,
        total_questions
    )


    return {

        "accuracy":
            round(accuracy * 100, 2),

        "precision":
            round(precision * 100, 2),

        "recall":
            round(recall * 100, 2)
    }


# ============================================================
# AVERAGE ANSWER CORRECTNESS
# ============================================================

def calculate_average_answer_correctness(
    results
):

    if not results:
        return 0.0

    total_score = sum(
        result["answer_correctness"]
        for result in results
    )

    average = (
        total_score
        /
        len(results)
    )

    return round(
        average * 100,
        2
    )