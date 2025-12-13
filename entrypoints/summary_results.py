import argparse
import asyncio
import os
import json

from evals import BENCHMARK_REGISTRY, BenchmarkFactory, BaseBenchmark


parser = argparse.ArgumentParser()
parser.add_argument(
    '--benchmark', 
    type=str, 
    required=True, 
    choices=list(BENCHMARK_REGISTRY.keys())
)
parser.add_argument(
    '--work_dir', 
    type=str, 
    required=True, 
)
parser.add_argument('--question_type', nargs='+', default=None)


async def main():
    args = parser.parse_args()
    work_dir = args.work_dir
    if not os.path.exists(work_dir):
        raise FileNotFoundError(f'args.work_dir ("{args.work_dir}") not found.')

    prediction_file = os.path.join(work_dir, 'predictions.jsonl')
    if not os.path.exists(prediction_file):
        raise FileNotFoundError(f'Prediction file ("{prediction_file}") not found.')
    
    predictions = {}
    with open(prediction_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    for line in lines:
        prediction_dict = json.loads(line)
        if (content := prediction_dict['content']) != '' and prediction_dict['sample_id'] not in predictions:
            predictions[prediction_dict['sample_id']] = content

    quesiton_type = None
    if args.question_type is None:
        with open(os.path.join(work_dir, 'config.json')) as f:
            config = json.load(f)
            quesiton_type = config['question_type']

    benchmark: BaseBenchmark = BenchmarkFactory.create_benchmark(
        benchmark_name=args.benchmark,
        question_type=quesiton_type,
    )
    benchmark.evaluate(predictions, output_dir=args.work_dir, ignore_empty=True)
    print(f'Evaluation finished. Results saved to: {os.path.abspath(args.work_dir)}')


if __name__ == '__main__':

    asyncio.run(main())
