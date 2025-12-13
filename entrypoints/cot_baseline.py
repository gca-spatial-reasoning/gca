import argparse
import asyncio
import json
import os
from typing import Dict

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from tqdm.asyncio import tqdm

from evals import (
    BaseBenchmark,
    BaseBenchmarkSample,
    BENCHMARK_REGISTRY, 
    BenchmarkFactory
)
from tools.apis import CoTReasoner, ImageBase64Encoder, ImageLoader
from workflow.config import AgentConfig


PROMPT_TEMPLATE = (
    "You are an expert spatial intelligence agent.\n\n"
    "{question}\n\n"
    "Please THINK STEP BY STEP. {data_specific_prompt}"
)

PROMPT_TEMPLATE_WITH_BBOX = (
    "You are an expert spatial intelligence agent.\n\n"
    "{question}\n\n"
    "The bounding boxes mentioned in the question are: {bbox}\n\n"
    "Please THINK STEP BY STEP. {data_specific_prompt}"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--benchmark', 
    type=str, 
    required=True, 
    choices=list(BENCHMARK_REGISTRY.keys())
)
parser.add_argument('--question_type', nargs='+', default=None)
parser.add_argument('--concurrency', type=int, default=1)
parser.add_argument('--work_dir', type=str, default=None)
parser.add_argument('--resume', action='store_true')


async def worker(
    reasoner: DeploymentHandle,
    image_loader: DeploymentHandle,
    benchmark: BaseBenchmark,
    sample: BaseBenchmarkSample,
    predictions: Dict,
    prediction_file: str,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock,
):
    async with semaphore:
        if benchmark.__class__.__name__ == 'MindCubeBench':
            sample_id = sample.sample_id
        else:
            sample_id = int(sample.sample_id)
        
        if hasattr(sample, 'bbox') and sample.bbox is not None: # for CVBench 3D
            prompt = PROMPT_TEMPLATE_WITH_BBOX.format(
                question=sample.question,
                bbox=sample.bbox,
                data_specific_prompt=benchmark.data_specific_prompt
            )
        else:
            prompt = PROMPT_TEMPLATE.format(
                question=sample.question,
                data_specific_prompt=benchmark.data_specific_prompt
            )

        try:
            load_image_refs = [
                image_loader.load_image.remote(image_source=image)
                for image in sample.images
            ]
            images_results = await asyncio.gather(*load_image_refs)
            loaded_images = [res['result'] for res in images_results if res.get('result')]

            result_ref = reasoner.cot_reason.remote(
                prompt=prompt, 
                input_images=loaded_images
            )
            result = await result_ref

            if result.get('err'):
                print(f'Error processing sample {sample_id}: {result["err"]}')
                prediction_content = ''
            else:
                prediction_content = result['result'].content
        except:
            prediction_content = ''

        async with lock:
            predictions[sample_id] = prediction_content
            
            saved_jsonl = {'sample_id': sample_id, 'content': prediction_content}
            with open(prediction_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(saved_jsonl) + '\n')


async def main():
    args = parser.parse_args()
    config = AgentConfig()
    config.update_from_args(args)

    if config.work_dir is None:
        model = config.cot_reasoner_model.replace('/', '--')
        config.work_dir = os.path.join(
            os.path.dirname(__file__), '..', 'work_dir', f'{config.benchmark}_baseline_{model}'
        )
    os.makedirs(config.work_dir, exist_ok=True)

    config_path = os.path.join(config.work_dir, 'config.json')
    os.environ['AGENT_CONFIG_FILE'] = config_path
    with open(config_path, 'w') as f:
        json.dump(config.to_json(), f, indent=4)

    # 1. Initialize Ray Serve and CoTReasoner
    image_loader = ImageLoader.bind()
    image_loader_handle = serve.run(image_loader, route_prefix='/image_loader')
    image_encoder = ImageBase64Encoder.bind(image_loader)
    reasoner = CoTReasoner.options(
        num_replicas=config.concurrency
    ).bind(image_encoder)
    reasoner_handle = serve.run(reasoner)

    # 2. Loading Benchmark
    benchmark: BaseBenchmark = BenchmarkFactory.create_benchmark(
        benchmark_name=config.benchmark,
        question_type=config.question_type,
    )

    # 3. Process Resume
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

    # 4. Dispatch Benchmark Samples
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
                reasoner=reasoner_handle,
                image_loader=image_loader_handle,
                benchmark=benchmark,
                sample=sample,
                predictions=predictions,
                prediction_file=prediction_file,
                semaphore=semaphore,
                lock=write_lock
            )
        ))

    # 5. Start Benchmarking
    print('Starting inference loop...')
    if tasks:
        await tqdm.gather(*tasks, desc=f'Evaluating {benchmark.__class__.__name__}')

    print('Inference complete. Shutting down Ray Serve...')
    serve.shutdown()
    ray.shutdown()

    # 6. Evaluating Predictions
    print('Evaluating predictions...')
    predictions = {sample.sample_id: predictions.get(sample.sample_id, '') for sample in benchmark}
    benchmark.evaluate(predictions, output_dir=config.work_dir)
    print(f'Evaluation finished. Results saved to: {os.path.abspath(config.work_dir)}')


if __name__ == '__main__':
    asyncio.run(main())
