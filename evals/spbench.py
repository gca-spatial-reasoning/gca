from dataclasses import dataclass
import json
import os
import re
from typing import Dict, List, Optional
from functools import partial

import pandas as pd
import numpy as np

from evals.base import BaseBenchmark, BaseBenchmarkSample


@dataclass
class SPBenchSample(BaseBenchmarkSample):
    is_numerical: bool
    subset: str


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    if pred is None or target is None:
        return 0.
    
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


class SPBench(BaseBenchmark):
    data_specific_prompt: str = (
        'Answer with a number if the question is about counting objects. '
        'Answer with the option\'s letter (A, B, C, D) if the question is multiple-choice, '
        'Answer should be enclosed in \\boxed{}.'
    )

    valid_question_types = [
        'MV_object_size_estimation',
        'MV_object_rel_direction',
        'MV_object_abs_distance',
        'MV_object_rel_distance',
        'MV_object_counting',
        'SI_object_size_estimation',
        'SI_object_rel_direction',
        'SI_object_abs_distance',
        'SI_object_rel_distance',
        'SI_object_counting',
    ]

    def __init__(self, data_path: str, question_type: Dict[str, List[str]] = None):
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
        self.eval_numerical = partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)

    def read_data(self):
        subset = set()
        for qt in self.question_type:
            if qt.startswith('MV'):
                subset.add('MV')
            if qt.startswith('SI'):
                subset.add('SI')
        subset = list(subset)
                
        data = None
        for subset_postfix in subset:
            data_subset_path = f'SPBench-{subset_postfix}'
            parquet_path = os.path.join(self.data_path, f'{data_subset_path}.parquet')
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f'Data file of SPBench not exists: {parquet_path}')

            image_dir = os.path.join(self.data_path, f'{data_subset_path}-images')
            if not os.path.exists(image_dir):
                raise FileNotFoundError(f'Image folder of SPBench not exists: {image_dir}')

            data_subset = pd.read_parquet(parquet_path)
            question_type_subset = [qt[3:] for qt in self.question_type if qt.startswith(subset_postfix)]
            data_subset = data_subset[data_subset['question_type'].isin(question_type_subset)]
            data_subset['subset'] = subset_postfix

            data = data_subset if data is None else pd.concat([data, data_subset])

        print('Evaluating question types:')
        for subset_postfix in subset:
            print(f'- {subset_postfix}')
            for qt in self.question_type:
                if qt.startswith(subset_postfix):
                    print(f'    {qt[3:]}')
        print(f'Totally {len(data)} samples.')

        image_paths = {}
        for _, row in data.iterrows():
            id_val, subset = row['id'], row['subset']

            images = row.get('images')
            if isinstance(images, np.ndarray):
                images = images.tolist()  # 转成普通 list
            elif images is None:
                images = []
            
            image_dir = os.path.join(self.data_path, f'SPBench-{row["subset"]}-images')
            image_paths[f'{subset}_{id_val}'] = [
                os.path.join(image_dir, row['scene_name'], img) for img in images
            ]

        return data, image_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> SPBenchSample:
        if index >= len(self.data):
            raise IndexError(f'Index {index} out of range (0-{len(self) - 1})')
        row = self.data.iloc[index]

        question = row['question']
        is_numerical = row['options'] is None
        if is_numerical:
            question += '<Numerical>'
        else:
            question += '<options>: '
            for option in row['options']:
                if option != 'can not determine':
                    question += option + '; '

        sid = f'{row["subset"]}_{row["id"]}'
        return SPBenchSample(
            sample_id=sid,
            images=self.image_paths.get(sid, []),
            question_type=row['question_type'],
            subset=row['subset'],
            question=question,
            answer=str(row['ground_truth']),
            is_numerical=is_numerical,
        )

    def extract_answer(self, prediction: str, is_numerical: bool) -> Optional[str]:
        if prediction is None:
            return None
        prediction = str(prediction).strip()

        match_boxed = re.search(r'\\boxed{\s*([A-D]|-?\d+(?:\.\d+)?)\s*}', prediction, re.IGNORECASE)
        if match_boxed:
            if is_numerical:
                return match_boxed.group(1)
            else:
                return match_boxed.group(1).upper()

        match = re.search(r'<\|begin_of_box\|>\s*([A-D]|-?\d+(?:\.\d+)?)\s*<\|end_of_box\|>', prediction, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        elif is_numerical:
            return None

        return None

    def evaluate(
        self,
        predictions: Dict[int | str, str],
        output_dir: Optional[str] = None,
        ignore_empty: bool = True,
    ) -> Dict:
        if ignore_empty:
            predictions = {k: v for k, v in predictions.items() if str(v).strip()}

        results = {
            'total_samples': 0,
            'score': 0,
            'overall_accuracy': 0.0,
            'detailed_results': [],
        }
        for subset in ['SI', 'MV']:
            results[subset] = {
                'total_samples': 0,
                'score': 0,
                'overall_accuracy': 0.0,
                'question_type_accuracy': {},
            }

        for _, row in self.data.iterrows():
            id_val = row['id']
            subset = row['subset']
            sid = f'{subset}_{id_val}'
            gt = str(row['ground_truth']).strip()
            qtype = row['question_type']

            pred_raw = predictions.get(sid, '')
            if ignore_empty and (not pred_raw or pred_raw.strip() == ''):
                continue

            if qtype not in results[subset]['question_type_accuracy']:
                results[subset]['question_type_accuracy'][qtype] = {'score': 0, 'total_samples': 0}

            is_numerical = row['options'] is None
            pred_extracted = self.extract_answer(pred_raw, is_numerical)
            if is_numerical:
                score = self.eval_numerical(float(pred_extracted), float(gt)) if pred_extracted else 0
            else:
                score = 1. if (pred_extracted == gt) else 0.

            results['total_samples'] += 1
            results[subset]['total_samples'] += 1
            results[subset]['question_type_accuracy'][qtype]['total_samples'] += 1
            results['score'] += score
            results[subset]['score'] += score
            results[subset]['question_type_accuracy'][qtype]['score'] += score

            results['detailed_results'].append({
                'id': sid,
                'question_type': qtype,
                'ground_truth': gt,
                'prediction': pred_raw,
                'extracted_answer': pred_extracted,
                'score': score,
            })

        results['overall_accuracy'] = (
            results['score'] / results['total_samples'] if results['total_samples'] else 0
        )

        for subset in ['SI', 'MV']:
            if results[subset]['total_samples'] == 0:
                del results[subset]
            else:
                results[subset]['overall_accuracy'] = (
                    results[subset]['score'] / results[subset]['total_samples']
                )
                for qt, s in results[subset]['question_type_accuracy'].items():
                    results[subset]['question_type_accuracy'][qt]['accuracy'] = (
                        s['score'] / s['total_samples']
                    )

        if output_dir:
            self.save_results(results, os.path.join(output_dir, 'spbench_results.csv'))
        self.pretty_print_results(results, output_dir)
        return results

    def pretty_print_results(self, results: Dict, output_dir: Optional[str] = None):
        print('\n' + '=' * 60)
        print('SPBench Evaluation Results')
        print('=' * 60)

        print(f'Total samples   : {results["total_samples"]:8.2f}')
        print(f'Score           : {results["score"]:8.2f}')
        print(f'Overall accuracy: {results["overall_accuracy"]:8.2%}')

        print('=' * 60)
        print('Accuracy by Question Type:')
        print('=' * 60)
        for subset_postfix in ['SI', 'MV']:
            if subset_postfix not in results:
                continue

            subset_results = results[subset_postfix]
            print(
                f'- {subset_postfix}: {subset_results["overall_accuracy"]:6.2%} '
                f'({subset_results["score"]:6.2f}/{subset_results["total_samples"]:6.2f})'
            )
            for qt, s in subset_results['question_type_accuracy'].items():
                print(
                    f'    {qt:30} {s["accuracy"]:6.2%} ({s["score"]:6.2f}/{s["total_samples"]:6.2f})'
                )
            print('=' * 60)

        summary = {
            'total_samples': int(results['total_samples']),
            'score': int(results['score']),
            'overall_accuracy': round(float(results['overall_accuracy']), 4),
            'subset': {
                str(s): {
                    'accuracy': round(float(results[s]['overall_accuracy']), 4),
                    'score': int(results[s]['score']),
                    'total_samples': int(results[s]['total_samples']),
                    'details': {
                        str(q): {
                            'accuracy': round(float(v['accuracy']), 4),
                            'score': int(v['score']),
                            'total_samples': int(v['total_samples']),
                        }
                        for q, v in results[s]['question_type_accuracy'].items()
                    }
                }
                for s in ['SI', 'MV'] if s in results
            },
        }

        if output_dir is not None:
            output_file = os.path.join(output_dir, 'results_summary.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
