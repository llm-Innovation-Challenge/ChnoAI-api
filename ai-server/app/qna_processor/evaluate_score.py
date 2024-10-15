from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def evaluate_bleu(reference, hypothesis):
    """reference, hypothesis 간의 BLEU 점수를 평가합니다."""    
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()

    smoothie = SmoothingFunction().method4
    score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)
    return score

def evaluate_rouge(reference, hypothesis):
    """reference, hypothesis 간의 ROUGE 점수를 평가합니다."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

def evaluate_recall(reference, hypothesis):
    """reference, hypothesis 간의  n-그램 중복을 기반으로 재현율(recall)을 계산합니다."""
    ref_ngrams = set(reference.split())
    hyp_ngrams = set(hypothesis.split())
    if len(ref_ngrams) == 0:
        return 0.0
    recall = len(ref_ngrams & hyp_ngrams) / len(ref_ngrams)
    return recall

def evaluate_processed_answer(original_answer, processed_answer):
    """원본 답변(original_answer)과 처리된 답변(processed_answer) 간의 BLEU, ROUGE, 재현율 점수를 평가합니다."""
    bleu_score = evaluate_bleu(original_answer, processed_answer)
    rouge_scores = evaluate_rouge(original_answer, processed_answer)
    recall_score = evaluate_recall(original_answer, processed_answer)
    result = {
        "bleu": bleu_score,
        "rouge": rouge_scores,
        "recall": recall_score
    }
    return result

def evaluate_coherence(original_question, summarized_question):
    """GEval Coherence를 사용하여 요약의 품질을 평가합니다."""
    if not original_question or not summarized_question:
        print("Invalid input for summarization. Original or summarized question is empty.")
        return {
            "coherence_score": None,
            "reason": "Invalid input: Empty question or summarized question"
        }


    test_case = LLMTestCase(input=original_question, actual_output=summarized_question)
    
    coherence_metric = GEval(
        name="Coherence",
        criteria="Coherence - determine if the actual output is coherent with the input, and summarizes the input text correctly.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        strict_mode=False,  
        verbose_mode=False,
        model="gpt-4o-mini"   
    )
    coherence_metric.measure(test_case)
    result = {
        "coherence_score" : coherence_metric.score,
        "reason" :coherence_metric.reason
    }
    return result

