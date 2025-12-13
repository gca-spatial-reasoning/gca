from dataclasses import dataclass
import json
import os
import re
import string
from typing import Dict, List, Optional

import pandas as pd
from evals.base import BaseBenchmark, BaseBenchmarkSample


@dataclass
class OmniSpatialSample(BaseBenchmarkSample):
    sub_question_type: str


class OmniSpatialBench(BaseBenchmark):
    data_specific_prompt: str = (
        'Answer with the option\'s letter (A, B, C, D). '
        'Enclose the option\'s letter within \\boxed{}.'
    )

    valid_question_types = [
        'Perspective_Taking',
        'Dynamic_Reasoning', 
        'Spatial_Interaction', 
        'Complex_Logic', 
    ]

    def __init__(self, data_path: str, question_type: List[str] = None):
        super().__init__()

        if question_type is None or 'all' in question_type:
            valid_types = self.valid_question_types
            invalid_types = []
        else:
            valid_types, invalid_types = [], []
            for qt in question_type:
                if qt in self.valid_question_types:
                    valid_types.append(qt)
                else:
                    invalid_types.append(qt)

        if not valid_types:
            raise ValueError(
                f'question_type {question_type} not supported. Expected {self.valid_question_types}.'
            )
        if invalid_types:
            print(
                f'[Warning] partial question_type {invalid_types} not supported. Expected '
                f'{self.valid_question_types}.'
            )

        self.question_type = valid_types
        self.data_path = data_path
        self.data, self.image_paths = self.read_data()

    def read_data(self) -> pd.DataFrame:
        json_path = os.path.join(self.data_path, 'data.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f'OmniSpatial data.json not found: {json_path}')

        with open(json_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)

        data = pd.DataFrame(data_json)
        data = data[data['task_type'].isin(self.question_type)]

        print('Evaluating question types:')
        for qt in self.question_type:
            print(qt)
        print(f'Totally {len(data)} samples.')

        image_paths = {}
        for idx, (_, row) in enumerate(data.iterrows()):
            image_num = str(row['id']).split('_')[0]
            image_path = os.path.join(self.data_path, row['task_type'], f'{image_num}.png')
            image_paths[idx] = [image_path] if os.path.exists(image_path) else []
        return data, image_paths

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> OmniSpatialSample:
        if index >= len(self.data):
            raise IndexError(f'Index {index} out of range (0-{len(self) - 1})')

        row = self.data.iloc[index]
        question=row['question'] + '<options>: '
        for idx, option in enumerate(row['options']):
            if option != 'can not determine':
             question += chr(65 + idx) + '.' + option + '; '

        return OmniSpatialSample(
            sample_id=index,
            images=self.image_paths.get(index),
            question_type=row['task_type'],
            sub_question_type=row.get('sub_task_type', ''),
            question=question,
            answer=chr(65 + int(row['answer'])),
        )

    def extract_answer(self, prediction: str, question_text: str) -> Optional[str]:
        """
        Extract multiple-choice question answers (A/B/C/D) from model predictions:
        1. LaTeX box format: \\boxed{A in answer text}, \\boxed{description of A in answer text}
        2. GLM-4.5V format: <|begin_of_box|>A<|end_of_box|>
        3. A., (A), A:
        4. When the cleaned string contains only a single valid letter

        Args:
            prediction (str): The model's original output string.
        
        Returns:
            The extracted answer ('A', 'B', 'C', 'D') or None.
        """
        if prediction is None:
            return None
        
        prediction = str(prediction).strip()

        def normalize_text(text):
            return text.lower().translate(str.maketrans('', '', string.punctuation + string.whitespace)).strip()
        
        options = {}
        # Pattern to find 'A: text, B: text, ...'
        option_matches = re.findall(r'([A-D]):\s*([^,.\n]*)', question_text, re.IGNORECASE)
        for key, value in option_matches:
            options[key.upper()] = value.strip()
        
        # 1: match \boxed{...}
        match = re.search(r'\\boxed{\s*([A-D])\s*}', prediction, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        if options:
            match = re.search(r'\\boxed{(.*)}', prediction, re.IGNORECASE)
            if match:
                boxed_text = match.group(1)
                normalized_boxed_text = normalize_text(boxed_text)
                for key, value in options.items():
                    normalized_option_text = normalize_text(value)
                    # Check if the normalized boxed text contains the normalized option text
                    if normalized_option_text in normalized_boxed_text:
                        return key.upper()

        # 2: match <|begin_of_box|>...<|end_of_box|>
        match = re.search(r'<\|begin_of_box\|>\s*([A-D])\s*<\|end_of_box\|>', prediction, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        # 3: match common format like A. (A) A:
        patterns = [
            r'\b([A-D])\.',
            r'\(([A-D])\)',
            r'\b([A-D]):',
        ]
        for pattern in patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # 4: Remove all punctuation and spaces, then check if only one valid answer letter remains.
        cleaned_prediction = prediction.translate(
            str.maketrans('', '', string.punctuation)
        ).replace(' ', '')
        if len(cleaned_prediction) == 1 and \
              cleaned_prediction.upper() in ['A', 'B', 'C', 'D']:
            return cleaned_prediction.upper()
            
        return None

    def evaluate(
        self,
        predictions: Dict[int | str, str],
        output_dir: Optional[str] = None,
        ignore_empty: bool = True,
    ) -> Dict:
        if ignore_empty:
            predictions = {sample_id: content for sample_id, content in predictions.items() if content.strip()}

        results = {
            'total_samples': 0,
            'correct_samples': 0,
            'overall_accuracy': 0.0,
            'detailed_results': [],
        }
        for subset in self.valid_question_types:
            results[subset] = {
                'total_samples': 0,
                'correct_samples': 0,
                'overall_accuracy': 0.0,
                'question_type_accuracy': {},
            }

        for idx, (_, row) in enumerate(self.data.iterrows()):
            gt = chr(65 + int(row['answer']))
            qtype = row['sub_task_type']
            subset = row['task_type']
            
            pred = predictions.get(idx, '')
            if ignore_empty and pred.strip() == '':
                continue

            if qtype not in results[subset]['question_type_accuracy']:
                results[subset]['question_type_accuracy'][qtype] = {'correct_samples': 0, 'total_samples': 0}

            extracted = self.extract_answer(pred, row['question'])
            is_correct = extracted == gt
            if is_correct:
                results['correct_samples'] += 1
                results[subset]['correct_samples'] += 1
                results[subset]['question_type_accuracy'][qtype]['correct_samples'] += 1
            results['total_samples'] += 1
            results[subset]['total_samples'] += 1
            results[subset]['question_type_accuracy'][qtype]['total_samples'] += 1

            results['detailed_results'].append({
                'id': idx,
                'question_type': row['sub_task_type'],
                'ground_truth': gt,
                'prediction': pred,
                'extracted_answer': extracted,
                'is_correct': is_correct,
            })

        results['overall_accuracy'] = results['correct_samples'] / results['total_samples']
        for subset in self.valid_question_types:
            if results[subset]['total_samples'] == 0:
                del results[subset]
            else:
                results[subset]['overall_accuracy'] = (
                    results[subset]['correct_samples'] / results[subset]['total_samples']
                )
                for qt, s in results[subset]['question_type_accuracy'].items():
                    results[subset]['question_type_accuracy'][qt]['accuracy'] = (
                        s['correct_samples'] / s['total_samples']
                    )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'results.csv')
            self.save_results(results, output_file)
        self.pretty_print_results(results, output_dir)
        return results

    def pretty_print_results(self, results: Dict, output_dir: Optional[str] = None):
        print('\n' + '=' * 60)
        print('OmniSpatial Evaluation Results')
        print('=' * 60)

        print(f'Total samples   : {results["total_samples"]:6d}')
        print(f'Correrct samples: {results["correct_samples"]:6d}')
        print(f'Overall accuracy: {results["overall_accuracy"]:6.2%}')
    
        print('=' * 60)
        print('Accuracy by Question Type:')
        print('=' * 60)
        for subset in self.valid_question_types:
            if subset not in results:
                continue

            subset_res = results[subset]
            print(
                f'- {subset}: {subset_res["overall_accuracy"]:7.2%} '
                f'({subset_res["correct_samples"]:3d}/{subset_res["total_samples"]:3d})'
            )
            for qt, s in subset_res['question_type_accuracy'].items():
                print(
                    f'    {qt:30} {s["accuracy"]:6.2%} ({s["correct_samples"]:3d}/{s["total_samples"]:3d})'
                )
            print('=' * 60)

        summary = {
            'total_samples': int(results['total_samples']),
            'correct_samples': int(results['correct_samples']),
            'overall_accuracy': round(float(results['overall_accuracy']), 4),
            'subset': {
                str(s): {
                    'accuracy': round(float(results[s]['overall_accuracy']), 4),
                    'correct_samples': int(results[s]['correct_samples']),
                    'total_samples': int(results[s]['total_samples']),
                    'details': {
                        str(q): {
                            'accuracy': round(float(v['accuracy']), 4),
                            'correct_samples': int(v['correct_samples']),
                            'total_samples': int(v['total_samples']),
                        }
                        for q, v in results[s]['question_type_accuracy'].items()
                    }
                }
                for s in self.valid_question_types
                if s in results
            }
        }

        if output_dir is not None:
            output_file = os.path.join(output_dir, 'results_summary.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
