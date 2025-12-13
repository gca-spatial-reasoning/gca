import argparse
import asyncio
import json
import os
import shutil
import traceback
import torch
from typing import Dict
from tqdm.asyncio import tqdm

from evals import (
    BaseBenchmark,
    BaseBenchmarkSample, 
    BENCHMARK_REGISTRY, 
    BenchmarkFactory
)
from workflow.config import AgentConfig
from workflow.state import AgentState
from workflow.workflow import AgentWorkflow


parser = argparse.ArgumentParser()
parser.add_argument(
    '--benchmark', 
    type=str, 
    required=True, 
    choices=list(BENCHMARK_REGISTRY.keys())
)
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--prompt_version', type=str, default='v1')
parser.add_argument('--question_type', nargs='+', default=None)
parser.add_argument('--concurrency', type=int, default=1)
parser.add_argument('--work_dir', type=str, default=None)
parser.add_argument('--resume', action='store_true')


async def worker(
    workflow: AgentWorkflow, 
    benchmark: BaseBenchmark,
    sample: BaseBenchmarkSample,
    predictions: Dict,
    prediction_file: str,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock
):
    async with semaphore:
        sample_id = sample.sample_id if isinstance(sample.sample_id, str) \
            else int(sample.sample_id)

        try:
            session_dir = workflow.logger.get_session_dir(str(sample_id))
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)

            instruction = f'{sample.question}\n\n{benchmark.data_specific_prompt}'
            bbox2d = None if (not hasattr(sample, 'bbox')) or sample.bbox is None \
                else torch.tensor(sample.bbox)
            final_state: AgentState = await workflow.arun(
                instruction=instruction,
                images=sample.images,
                bbox2d=bbox2d,
                answer=sample.answer,
                session_id=str(sample_id)
            )
            _, summary = workflow.get_final_answer(final_state)
        except Exception as e:
            print(f'[Error] {str(e)}')
            print(traceback.format_exc())
            summary = ''

        async with lock:
            predictions[sample_id] = summary

            saved_jsonl = {'sample_id': sample_id, 'content': summary}
            with open(prediction_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(saved_jsonl) + '\n')


async def main():
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', f'agent_{args.benchmark}.json'
        )

    config = AgentConfig()
    config.update_from_json(config_path)
    config.update_from_args(args)

    if config.work_dir is None:
        model = config.cot_reasoner_model.replace('/', '--')
        prompt = config.prompt_version
        config.work_dir = os.path.join(
            os.path.dirname(__file__), '..', 'work_dir', f'{config.benchmark}_{model}_{prompt}'
        )
    os.makedirs(config.work_dir, exist_ok=True)

    config_path = os.path.join(config.work_dir, 'config.json')
    os.environ['AGENT_CONFIG_FILE'] = config_path
    with open(config_path, 'w') as f:
        json.dump(config.to_json(), f, indent=4)

    # 1. Initialize Benchmark
    benchmark: BaseBenchmark = BenchmarkFactory.create_benchmark(
        benchmark_name=config.benchmark,
        question_type=config.question_type,
    )

    # Process Resume
    predictions, done = {}, set()
    prediction_file = os.path.join(config.work_dir, 'predictions.jsonl')
    if args.resume and os.path.exists(prediction_file):
        with open(prediction_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        for line in lines:
            prediction_dict = json.loads(line)
            content = prediction_dict['content']
            if content != '' and content is not None:
                predictions[prediction_dict['sample_id']] = content
                done.add(prediction_dict['sample_id'])

        with open(prediction_file, 'w') as f:
            f.writelines([
                json.dumps(dict(sample_id=sample_id, content=content)) + '\n' 
                for sample_id, content in predictions.items()
            ])
        print(f'Resuming benchmarking. Found {len(done)} completed samples.')
    else:
        with open(prediction_file, 'w') as f:
            pass

    # 2. Initialize AgentWorkflow
    workflow = AgentWorkflow()

    # 3. Dispatch Benchmark Samples
    concurrency = min(args.concurrency, len(benchmark))
    print(f'Executing tasks with concurrency={concurrency}')
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    tasks = []
    for sample in benchmark:
        if sample.sample_id in done:
            continue        
        tasks.append(asyncio.create_task(
            worker(
                workflow=workflow,
                benchmark=benchmark,
                sample=sample,
                predictions=predictions,
                prediction_file=prediction_file,
                semaphore=semaphore,
                lock=write_lock,
            )
        ))

    # 4. Start Benchmarking
    print('Starting inference loop...')
    if tasks:
        await tqdm.gather(*tasks, desc=f'Evaluating {benchmark.__class__.__name__}')

    print('Inference complete. Shutting down Ray Serve...')
    workflow.shutdown()

    # 5. Evaluating Predictions
    print('Evaluating predictions...')
    predictions = {sample.sample_id: predictions.get(sample.sample_id, '') for sample in benchmark}
    benchmark.evaluate(predictions, output_dir=config.work_dir)
    print(f'Evaluation finished. Results saved to: {os.path.abspath(config.work_dir)}')


if __name__ == '__main__':
    asyncio.run(main())
