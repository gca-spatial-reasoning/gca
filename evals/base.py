from dataclasses import dataclass
import json
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class BaseBenchmarkSample:
    sample_id: int | str
    question: str
    question_type: str
    images: List[str]
    answer: str


class BaseBenchmark:
    data_specific_prompt: str = ''

    def __init__(self):
        self.n = 0

    def __getitem__(self, index: int) -> BaseBenchmarkSample:
        pass

    def __len__(self) -> int:
        pass

    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self) -> BaseBenchmarkSample:
        if self.n < len(self):
            sample = self[self.n]
            self.n += 1
            return sample
        else:
            raise StopIteration

    def extract_answer(self, prediction: str) -> Optional[str]:
        pass

    def evaluate(
        self, 
        predictions: List[str] | Dict[int, str],
        output_dir: str = None,
        ignore_empty: bool = False,
    ):
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: list aligned with data rows, or dict {id: prediction}
            output_dir: optional path to save detailed results
        
        Returns:
            Dict with overall and per-type accuracy and detailed rows
        """
        pass

    def pretty_print_results(results: Dict, output_dir: Optional[str] = None):
        pass

    def save_results(self, results: Dict, output_file: str):
        detailed_df = pd.DataFrame(results['detailed_results'])
        if output_file.endswith('.xlsx'):
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                detailed_df.to_excel(writer, sheet_name='Details', index=False)
        elif output_file.endswith('.csv'):
            detailed_df.to_csv(output_file, index=False)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
