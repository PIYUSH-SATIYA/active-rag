import json
import os
from typing import Dict, List, Any

class WikiMultiHopQA:
    def __init__(self, data_dir: str = "data/dummy", split: str = "dev"):
        self.data_dir = data_dir
        self.split = split
        self.data_file = os.path.join(data_dir, f"{split}.json")
        self.alias_file = os.path.join(data_dir, "id_aliases.json")
        
        self.examples = self._load_data()
        self.aliases = self._load_aliases()

    def _load_data(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.data_file):
            return []
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        formatted_data = []
        for item in data:
            formatted_data.append({
                "id": item.get("_id", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "type": item.get("type", ""),
                "gold_contexts": item.get("context", []),
                "supporting_facts": item.get("supporting_facts", [])
            })
        return formatted_data

    def _load_aliases(self) -> Dict[str, List[str]]:
        if not os.path.exists(self.alias_file):
            return {}
            
        with open(self.alias_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
        
    def get_aliases(self, example_id: str) -> List[str]:
        return self.aliases.get(example_id, [])

    @staticmethod
    def normalize_answer(s: str) -> str:
        import string
        import re
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(cls, prediction: str, ground_truth: str, ground_truth_id: str = None) -> Dict[str, float]:
        norm_pred = cls.normalize_answer(prediction)
        norm_truth = cls.normalize_answer(ground_truth)
        correct = int(norm_pred == norm_truth)
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(cls, prediction: str, ground_truth: str, ground_truth_id: str = None) -> Dict[str, float]:
        norm_pred = cls.normalize_answer(prediction)
        norm_truth = cls.normalize_answer(ground_truth)
        
        pred_tokens = norm_pred.split()
        truth_tokens = norm_truth.split()
        
        from collections import Counter
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
            
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return {'f1': f1, 'precision': precision, 'recall': recall}

