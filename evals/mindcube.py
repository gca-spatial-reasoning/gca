from dataclasses import dataclass
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd

from evals.base import BaseBenchmark, BaseBenchmarkSample


@dataclass
class MindCubeBenchSample(BaseBenchmarkSample):
    category: List[str]
    sample_type: str


class MindCubeBench(BaseBenchmark):
    
    data_specific_prompt: str = (
        "Based on these images, answer the question based on this rule: You only need to provide\n"
        "*ONE* correct answer selecting from the options listed below. For example, if you think\n"
        "the correct answer is 'A. above' from ' A. above B. under C. front D. behind.', your\n"
        "response should only be 'A. above'."
    )
    
    valid_question_types = [
        'among',
        'around',
        'rotation'
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
        self.data, self.image_base_dir = self.read_data()

    def read_data(self):
        jsonl_files = ['MindCube_tinybench.jsonl']
        
        search_dirs = [
            self.data_path,
            os.path.join(self.data_path, 'raw'),
        ]
        
        jsonl_path = None
        for search_dir in search_dirs:
            for filename in jsonl_files:
                path = os.path.join(search_dir, filename)
                if os.path.exists(path):
                    jsonl_path = path
                    break
            if jsonl_path:
                break
        
        if jsonl_path is None:
            raise FileNotFoundError(
                f'MindCube data file not found in: {self.data_path}\n'
                f'Searched in: {search_dirs}\n'
                f'Expected one of: {jsonl_files}'
            )
        
        data_list = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
        
        data = pd.DataFrame(data_list)
        data['scene_type'] = data['id'].apply(self._get_scene_type_static)
        data = data[data['scene_type'].isin(self.question_type)]
        
        print('Evaluating question types:')
        for qt in self.question_type:
            print(f'  - {qt}')
        print(f'Totally {len(data)} samples.')
        
        image_base_dir = os.path.join(self.data_path, 'images')
        if not os.path.exists(image_base_dir):
            alt_paths = [
                os.path.join(self.data_path, 'other_all_image'),
                os.path.join(os.path.dirname(self.data_path), 'other_all_image'),
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    image_base_dir = os.path.dirname(alt_path)
                    break
        
        return data, image_base_dir

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> MindCubeBenchSample:
        if index >= len(self.data):
            raise IndexError(f'Index {index} out of range (0-{len(self) - 1})')
        
        row = self.data.iloc[index]
        
        images = row['images']
        if isinstance(images, str):
            images = json.loads(images)
        
        image_paths = []
        for img in images:
            if not os.path.isabs(img):
                img_path = os.path.join(self.image_base_dir, img)
            else:
                img_path = img
            image_paths.append(img_path)
        
        scene_type = self._get_scene_type(row['id'])
        
        return MindCubeBenchSample(
            sample_id=row['id'],
            question=row['question'],
            images=image_paths,
            answer=row['gt_answer'],
            category=row.get('category', []),
            sample_type=row.get('type', ''),
            question_type=scene_type 
        )

    @staticmethod
    def _get_scene_type_static(sample_id: str) -> str:
        sample_id_lower = str(sample_id).lower()
        
        if 'around' in sample_id_lower:
            return 'around'
        elif 'rotation' in sample_id_lower:
            return 'rotation'
        elif 'translation' in sample_id_lower:
            return 'translation'
        elif 'among' in sample_id_lower:
            return 'among'
        else:
            return 'other'
    
    def _get_scene_type(self, sample_id: str) -> str:
        return self._get_scene_type_static(sample_id)

    def extract_answer(self, prediction: str, question_text: str = '') -> Optional[str]:
        """
        Extract answer from model prediction.
        
        Supports multiple formats (priority order):
        1. LaTeX box: \\boxed{A}
        2. Simple format: A., B., C.
        3. Tag format: <Answer>A</Answer> or <answer>A</answer>
        4. Text format: "My answer is A", "The answer is B"
        5. Single letter: A, B, C, D, E
        
        Args:
            prediction: Model's prediction text
            question_text: Original question (unused, for compatibility)
            
        Returns:
            Extracted answer letter or None
        """
        if prediction is None:
            return None
        
        prediction = str(prediction).strip()
        
        match = re.search(r'\\boxed{\s*([A-E])\s*}', prediction, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        simple_matches = list(re.finditer(r'\b([A-E])\.', prediction))
        if simple_matches:
            return simple_matches[-1].group(1).upper()
        
        for tag in ['Answer', 'answer']:
            answer_match = re.search(f'<{tag}>(.*?)</{tag}>', prediction, re.DOTALL)
            if answer_match:
                answer_section = answer_match.group(1)
                for pattern in [
                    r'\b([A-E])\.',
                    r'\b([A-E])\b',
                ]:
                    matches = list(re.finditer(pattern, answer_section))
                    if matches:
                        return matches[-1].group(1).upper()
        
        # Priority 4: Text patterns
        text_patterns = [
            r'[Mm]y answer is ([A-E])',
            r'[Tt]he answer is ([A-E])',
            r'(?:Answer|answer):\s*([A-E])',
            r'\b([A-E])\s*[:.]\s*[A-Z]',  # A: Some text or A. Some text
        ]
        
        for pattern in text_patterns:
            matches = list(re.finditer(pattern, prediction))
            if matches:
                return matches[-1].group(1).upper()
        
        # Priority 5: Single letter (last occurrence)
        single_letter_matches = list(re.finditer(r'\b([A-E])\b', prediction))
        if single_letter_matches:
            return single_letter_matches[-1].group(1).upper()
        
        return None

    def evaluate(
        self, 
        predictions: Dict[str, str],
        output_dir: Optional[str] = None,
        ignore_empty: bool = False,
    ) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List aligned with data or dict {sample_id: prediction}
            output_dir: Optional directory to save results
            ignore_empty: Whether to ignore empty predictions
        Returns:
            Dictionary with evaluation results
        """
        # 
        if ignore_empty:
            predictions = {sample_id: content for sample_id, content in predictions.items() if content.strip()}

        results = {
            'total_samples': 0,
            'correct_samples': 0,
            'overall_accuracy': 0.0,
            'detailed_results': [],
            'question_type_accuracy': {},
        }
        
        for _, row in self.data.iterrows():
            sample_id = row['id']
            ground_truth = str(row['gt_answer']).strip().upper()
            qtype = self._get_scene_type(sample_id)

            prediction = predictions.get(sample_id, '')
            if ignore_empty and prediction.strip() == '':
                continue
            if qtype == 'translation':
                continue

            if qtype not in results['question_type_accuracy']:
                results['question_type_accuracy'][qtype] = {'correct_samples': 0, 'total_samples': 0}
            
            extracted_answer = self.extract_answer(prediction, row['question'])
            is_correct = (extracted_answer == ground_truth) if extracted_answer else False
            if is_correct:
                results['correct_samples'] += 1
                results['question_type_accuracy'][qtype]['correct_samples'] += 1
            results['total_samples'] += 1
            results['question_type_accuracy'][qtype]['total_samples'] += 1

            results['detailed_results'].append({
                'id': sample_id,
                'question_type': qtype,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct
            })
        
        results['overall_accuracy'] = results['correct_samples'] / results['total_samples']
        for qt, s in results['question_type_accuracy'].items():
            if s['total_samples'] == 0:
                del results['question_type_accuracy'][qt]
            else:
                results['question_type_accuracy'][qt]['accuracy'] = (
                    s['correct_samples'] / s['total_samples']
                )
        
        # Save and print results
        if output_dir is not None:
            output_file = os.path.join(output_dir, 'results.csv')
            self.save_results(results, output_file)
        self.pretty_print_results(results, output_dir)
        return results

    def pretty_print_results(self, results: Dict, output_dir: Optional[str] = None):
        print('\n' + '=' * 60)
        print('MindCube Evaluation Results')
        print('=' * 60)
        
        print(f'Total samples   : {results["total_samples"]:6d}')
        print(f'Correct samples : {results["correct_samples"]:6d}')
        print(f'Overall accuracy: {results["overall_accuracy"]:6.2%}')
        
        print('=' * 60)
        print('Accuracy by Question Type:')
        print('=' * 60)
        for qt, s in results['question_type_accuracy'].items():
            print(
                f'{qt:20} {s["accuracy"]:6.2%} ({s["correct_samples"]:3d}/{s["total_samples"]:3d})'
            )
        print('=' * 60)
        
        # Save JSON summary
        if output_dir is not None:
            summary = {
                'total_samples': int(results['total_samples']),
                'correct_samples': int(results['correct_samples']),
                'overall_accuracy': round(float(results['overall_accuracy']), 4),
                'question_type_accuracy': {
                    str(qt): {
                        'accuracy': round(float(s['accuracy']), 4),
                        'correct_samples': int(s['correct_samples']),
                        'total_samples': int(s['total_samples']),
                    }
                    for qt, s in results['question_type_accuracy'].items()
                }
            }
            
            output_file = os.path.join(output_dir, 'results_summary.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f'Summary saved to: {output_file}')
