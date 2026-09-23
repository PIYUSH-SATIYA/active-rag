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
            aliases_dict = {}
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    aliases_dict[item["Q_id"]] = item.get("aliases", [])
            return aliases_dict

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
        if precision + recall == 0:
            f1 = 0.0
        return {'f1': f1, 'precision': precision, 'recall': recall}


class StrategyQA(WikiMultiHopQA):
    """
    Commonsense reasoning dataset.
    Gold answers are 'true' / 'false'; the model tends to output 'yes' / 'no',
    so exact_match_score normalises both directions.
    """
    def __init__(self, data_dir: str = "data/strategyqa", split: str = "dev"):
        super().__init__(data_dir, split)

    def _load_data(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.data_file):
            print(f"Warning: Data file {self.data_file} not found. Returning empty dataset.")
            return []
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        formatted_data = []
        for item in data:
            formatted_data.append({
                "id": item.get("qid", item.get("id", "")),
                "question": item.get("question", ""),
                "answer": str(item.get("answer", "")).lower(),
            })
        return formatted_data

    @staticmethod
    def _normalise_bool(text: str) -> str:
        """Map yes→true and no→false so model output matches gold labels."""
        t = text.lower().strip()
        if t in ("yes", "true"):
            return "true"
        if t in ("no", "false"):
            return "false"
        return t

    @classmethod
    def exact_match_score(cls, prediction: str, ground_truth: str,
                          ground_truth_id: str = None) -> Dict[str, float]:
        norm_pred  = cls._normalise_bool(cls.normalize_answer(prediction))
        norm_truth = cls._normalise_bool(cls.normalize_answer(ground_truth))
        correct = int(norm_pred == norm_truth)
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(cls, prediction: str, ground_truth: str,
                 ground_truth_id: str = None) -> Dict[str, float]:
        # For binary yes/no, F1 == EM — reuse the normalised exact match
        em = cls.exact_match_score(prediction, ground_truth)
        score = float(em['correct'])
        return {'f1': score, 'precision': score, 'recall': score}


class ASQA(WikiMultiHopQA):
    """
    Ambiguous Short Question Answering dataset.

    Gold answers come as a *list* of acceptable short answers collected from
    qa_pairs[*].short_answers.  We store the full list and override the
    scoring methods to accept any match — the same convention used in the
    original ASQA evaluation script.

    The dev.json written by the Colab downloader has the field
    "short_answers": ["ans1", "ans2", ...] at the top level.
    """
    def __init__(self, data_dir: str = "data/asqa", split: str = "dev"):
        super().__init__(data_dir, split)

    def _load_data(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.data_file):
            print(f"Warning: Data file {self.data_file} not found. Returning empty dataset.")
            return []
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        formatted_data = []
        for item in data:
            # Support both the new downloader format (short_answers list) and
            # the raw HuggingFace format (qa_pairs list) for robustness.
            if "short_answers" in item:
                answers = item["short_answers"]  # already flattened by downloader
            else:
                # Flatten short_answers from every qa_pair entry
                answers = []
                for qa_pair in item.get("qa_pairs", []):
                    answers.extend(qa_pair.get("short_answers", []))
            # Deduplicate while preserving order
            seen: set = set()
            unique_answers = []
            for a in answers:
                if a not in seen:
                    seen.add(a)
                    unique_answers.append(a)

            formatted_data.append({
                "id": str(item.get("sample_id", item.get("id", ""))),
                "question": item.get("ambiguous_question", item.get("question", "")),
                # Store as newline-joined string so the base runner works unchanged;
                # scoring methods below split it back into a list.
                "answer": "\n".join(unique_answers),
            })
        return formatted_data

    @classmethod
    def exact_match_score(cls, prediction: str, ground_truth: str,
                          ground_truth_id: str = None) -> Dict[str, float]:
        """Return 1 if prediction matches *any* of the gold short answers."""
        norm_pred = cls.normalize_answer(prediction)
        gold_answers = [cls.normalize_answer(a)
                        for a in ground_truth.split("\n") if a.strip()]
        if not gold_answers:
            return {'correct': 0, 'incorrect': 1}
        correct = int(norm_pred in gold_answers)
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(cls, prediction: str, ground_truth: str,
                 ground_truth_id: str = None) -> Dict[str, float]:
        """Return the best token-F1 across all gold short answers."""
        from collections import Counter
        norm_pred = cls.normalize_answer(prediction)
        pred_tokens = norm_pred.split()
        gold_answers = [cls.normalize_answer(a)
                        for a in ground_truth.split("\n") if a.strip()]
        if not gold_answers or not pred_tokens:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

        best_f1 = best_prec = best_rec = 0.0
        for gold in gold_answers:
            truth_tokens = gold.split()
            common = Counter(pred_tokens) & Counter(truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            prec = num_same / len(pred_tokens)
            rec  = num_same / len(truth_tokens)
            f1   = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1, best_prec, best_rec = f1, prec, rec
        return {'f1': best_f1, 'precision': best_prec, 'recall': best_rec}


class WikiAsp(WikiMultiHopQA):
    """
    Open-domain summarization dataset.

    The Colab downloader stores each example as:
        {"id": exid, "topic": <entity name>, "summary": <first reference summary>}

    "topic" is used as the question (the thing to summarise) and
    "summary" is the gold reference text used for ROUGE-L evaluation.
    Standard EM/F1 are a rough proxy here; the paper uses ROUGE-L / UniEval.
    """
    def __init__(self, data_dir: str = "data/wikiasp", split: str = "dev"):
        super().__init__(data_dir, split)

    def _load_data(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.data_file):
            print(f"Warning: Data file {self.data_file} not found. Returning empty dataset.")
            return []
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        formatted_data = []
        for item in data:
            # Resolve topic / question ------------------------------------------
            # New downloader format: "topic" is a plain string (entity name).
            # Raw HF parquet format: "input" is a list of passage strings whose
            # first line is the entity name.
            topic = item.get("topic", "")
            if not topic:
                raw_input = item.get("input", "")
                if isinstance(raw_input, list) and raw_input:
                    # first line of first passage = entity name
                    topic = raw_input[0].split("\n")[0].strip()
                elif isinstance(raw_input, str) and raw_input:
                    topic = raw_input.split("\n")[0].strip()

            # Resolve summary / answer ------------------------------------------
            # New downloader: "summary" is a plain string.
            # Raw HF parquet: "output" is a list of reference summaries.
            raw_output = item.get("summary", item.get("output", ""))
            if isinstance(raw_output, list):
                answer = raw_output[0] if raw_output else ""
            else:
                answer = raw_output

            formatted_data.append({
                "id": item.get("id", ""),
                "question": topic,
                "answer": answer,
            })
        return formatted_data


def get_dataset(name: str, data_dir: str, split: str = "dev"):
    """
    Factory function to retrieve the appropriate dataset class.
    """
    name = name.lower().replace("-", "").replace("_", "")
    if name in ["2wikihop", "2wikimultihopqa", "wikimultihopqa"]:
        return WikiMultiHopQA(data_dir, split)
    elif name in ["strategyqa", "strategy"]:
        return StrategyQA(data_dir, split)
    elif name == "asqa":
        return ASQA(data_dir, split)
    elif name in ["wikiasp", "wikiasp", "wikilsp"]:
        return WikiAsp(data_dir, split)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
