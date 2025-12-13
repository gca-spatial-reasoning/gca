from dataclasses import dataclass
import io
import json
import os
import re
import string
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
import tqdm

from evals.base import BaseBenchmark, BaseBenchmarkSample


@dataclass
class CVBenchSample(BaseBenchmarkSample):
    bbox: List


class CVBench(BaseBenchmark):
    data_specific_prompt: str = '''
** Crucial Rules for CVBench**
1. For questions involving bounding boxes, please directly use the `input_bbox2d` from workspace. Do not call the detection tool again to find the bounding box. Specifically, please determine the index of the bounding box you need to use by referring to the image and the [Bounding Box Value]. The bounding box indices in the image correspond to the order in the value list. For example, if you need to pass the 0th box, please use `$input_bbox2d.bbox2d[0]` directly.
2. If there is no explicit description to clarify the coordinate system, please directly use the camera frame as the reference frame, namely `+X_ref=+X_cam=right, +Y_ref=+Y_cam=down, +Z_ref=+Z_cam=front`
3. Answer with the option's letter (A, B, C, D, ...). Enclose the option's letter within \\boxed{}.
    '''

    valid_question_types = [
        'Count', 'Depth', 'Distance', 'Relation'
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

    def dump_images(self):
        def _dump_parquet(parquet_path: str):
            df = pd.read_parquet(parquet_path)
            for _, row in tqdm.tqdm(
                df.iterrows(), total=len(df), desc=f'Dumping images from parquet "{parquet_path}" ...'
            ):
                img_bytes = row['image']['bytes']
                img = Image.open(io.BytesIO(img_bytes))

                dir_path = os.path.dirname(row['filename'])
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                img.save(row['filename'])

        parquet_2d = os.path.join(self.data_path, 'test_2d.parquet')
        _dump_parquet(parquet_path=parquet_2d)
        parquet_3d = os.path.join(self.data_path, 'test_3d.parquet')
        _dump_parquet(parquet_path=parquet_3d)

    def read_data(self) -> pd.DataFrame:
        json_path = os.path.join(self.data_path, 'data.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f'CV-Bench data.json not found: {json_path}')

        with open(json_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        data = pd.DataFrame(data_json)

        img_dir = os.path.join(self.data_path, 'img')
        if not os.path.exists(img_dir):
            self.dump_images()
        
        data = data[data['task'].isin(self.question_type)]
        print('Evaluating question types:')
        for qt in self.question_type:
            print(qt)
        print(f'Totally {len(data)} samples.')

        image_paths = {}
        for idx, (_, row) in enumerate(data.iterrows()):
            image_path = os.path.join(self.data_path, row['filename'])
            image_paths[idx] = [image_path] if os.path.exists(image_path) else []

        return data, image_paths

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> CVBenchSample:
        if index >= len(self.data):
            raise IndexError(f'Index {index} out of range (0-{len(self) - 1})')

        row = self.data.iloc[index]
        return CVBenchSample(
            sample_id=index,
            images=self.image_paths.get(index),
            question_type=row['task'],
            question=row['prompt'],
            answer=row['answer'],
            bbox=row['bbox']
        )
    
    def get_subset(self, question_type: str) -> str:
        if question_type in ['Count', 'Relation']:
            return '2D'
        else:
            return '3D'

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
        option_matches = re.findall(r'([A-F]):\s*([^,.\n]*)', question_text, re.IGNORECASE)
        for key, value in option_matches:
            options[key.upper()] = value.strip()
        
        # 1: match \boxed{...}
        match = re.search(r'\\boxed{\s*([A-F])\s*}', prediction, re.IGNORECASE)
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
        match = re.search(r'<\|begin_of_box\|>\s*([A-F])\s*<\|end_of_box\|>', prediction, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        # 3: match common format like A. (A) A:
        patterns = [
            r'\b([A-F])\.',
            r'\(([A-F])\)',
            r'\b([A-F]):',
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
              cleaned_prediction.upper() in ['A', 'B', 'C', 'D', 'E', 'F']:
            return cleaned_prediction.upper()
            
        return None

    def evaluate(
        self,
        predictions: List[str] | Dict[int, str],
        output_dir: Optional[str] = None,
        ignore_empty: bool = False,
    ) -> Dict:
        if ignore_empty:
            predictions = {sample_id: content for sample_id, content in predictions.items() if content.strip()}

        results = {
            'total_samples': 0,
            'correct_samples': 0,
            'overall_accuracy': 0.0,
            'detailed_results': [],
        }
        for subset in ['2D', '3D']:
            results[subset] = {
                'total_samples': 0,
                'correct_samples': 0,
                'overall_accuracy': 0.0,
                'question_type_accuracy': {},
            }

        for idx, (_, row) in enumerate(self.data.iterrows()):
            ground_truth = str(row['answer'][1]).strip().upper()
            qtype = row['task']
            subset = self.get_subset(qtype)
            
            pred = predictions.get(idx, '')
            if ignore_empty and pred.strip() == '':
                continue

            if qtype not in results[subset]['question_type_accuracy']:
                results[subset]['question_type_accuracy'][qtype] = {'correct_samples': 0, 'total_samples': 0}

            extracted_answer = self.extract_answer(pred, row['question'])
            is_correct = (extracted_answer == ground_truth) if extracted_answer else False
            
            results['total_samples'] += 1
            results[subset]['total_samples'] += 1
            results[subset]['question_type_accuracy'][qtype]['total_samples'] += 1
            if is_correct:
                results['correct_samples'] += 1
                results[subset]['correct_samples'] += 1
                results[subset]['question_type_accuracy'][qtype]['correct_samples'] += 1

            results['detailed_results'].append({
                'id': idx,
                'subset': subset,
                'question_type': qtype,
                'ground_truth': ground_truth,
                'prediction': pred,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
            })

        results['overall_accuracy'] = results['correct_samples'] / results['total_samples']
        for subset in ['2D', '3D']:
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
            output_file = os.path.join(output_dir, 'results.csv')
            self.save_results(results, output_file)
        self.pretty_print_results(results, output_dir)
        return results

    def pretty_print_results(self, results: Dict, output_dir: Optional[str] = None):
        print('\n' + '=' * 60)
        print('CVBench Evaluation Results')
        print('=' * 60)

        print(f'Total samples   : {results["total_samples"]:6d}')
        print(f'Correct samples : {results["correct_samples"]:6d}')
        print(f'Overall accuracy: {results["overall_accuracy"]:6.2%}')

        print('=' * 60)
        print('Accuracy by Question Type:')
        print('=' * 60)
        for subset in ['2D', '3D']:
            if subset not in results:
                continue

            subset_res = results[subset]
            print(
                f'- {subset}: {subset_res["overall_accuracy"]:7.2%} '
                f'({subset_res["correct_samples"]:3d}/{subset_res["total_samples"]:3d})'
            )
            for qt, s in subset_res['question_type_accuracy'].items():
                print(
                    f'    {qt:20} {s["accuracy"]:6.2%} ({s["correct_samples"]:3d}/{s["total_samples"]:3d})'
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
                for s in ['2D', '3D'] 
                if s in results
            }
        }

        if output_dir is not None:
            output_file = os.path.join(output_dir, 'results_summary.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
