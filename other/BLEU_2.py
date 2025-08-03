import sacrebleu

class Evaluator:
    def __init__(self):
        """仅保留BLEU计算功能"""
        pass
        
    def calculate_bleu(self, hypothesis, references):
        """使用SacreBLEU计算BLEU分数"""
        if isinstance(references[0], list):
            refs = [[ref] if isinstance(ref, str) else ref for ref in references]
        else:
            refs = [references]
        return sacrebleu.corpus_bleu(hypothesis, refs).score
    
    def evaluate_all(self, hypothesis, references, src_text=None):
        """仅评估BLEU"""
        return {
            "bleu": self.calculate_bleu(hypothesis, references),
            "src_text": src_text
        }