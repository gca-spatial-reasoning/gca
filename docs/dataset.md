# Evaluation Dataset

## MMSI-Bench

Download the dataset from 🤗 [Hugging Face](https://huggingface.co/datasets/RunsenXu/MMSI-Bench/tree/main).

The file structure will be as follow:

```
gca
├── ...
├── data
│   ├── ...
│   ├── mmsi
│   │   ├── MMSI_Bench.parquet
│   │   └── images # After the first run of the code, it will be automatically created.
│   │       ├── 0_0.jpg
│   │       ├── ...
│   ├── ...
├── ...
```

## MindCube

Download the dataset from 🤗 [Hugging Face](https://huggingface.co/datasets/MLL-Lab/MindCube).

Unzip the `data.zip`

```bash
unzip data.zip

mkdir -p gca/data/mindcube
mv raw other_all_image gca/data/mindcube
```

The file structure will be as follow:

```
gca
├── ...
├── data
│   ├── ...
│   ├── mindcube
│   │   ├── raw
│   │   │   ├── MindCube.jsonl
│   │   │   ├── MindCube_train.jsonl
│   │   │   └── MindCube_tinybench.jsonl
│   │   └── other_all_image
│   │       ├── around
│   │       ├── among
│   │       └── rotation
│   ├── ...
├── ...
```

## OmniSpatial

Download the dataset from 🤗 [Hugging Face](https://huggingface.co/datasets/qizekun/OmniSpatial).

Unzip the `OmniSpatial-test.zip`.

```bash
unzip OmniSpatial-test.zip
mv OmniSpatial-test gca/data/omnispatial
```

The file structure will be as follow:

```
gca
├── ...
├── data
│   ├── ...
│   ├── omnispatial
│   │   ├── data.json
│   │   ├── Complex_Logic
│   │   │   ├── 1.png
│   │   │   ├── ...
│   │   ├── Dynamic_Reasoning
│   │   │   ├── 1.png
│   │   │   ├── ...
│   │   ├── ...
│   ├── ...
├── ...
```

## SPBench

Download the dataset from 🤗 [Hugging Face](https://huggingface.co/datasets/hongxingli/SPBench).

Unzip the images archive.

```bash
unzip SPBench-MV-images.zip
unzip SPBench-SI-images.zip

mkdir -p gca/data/spbench
mv SPBench-MV-images SPBench-SI-images SPBench-MV.parquet SPBench-SI.parquet gca/data/spbench
```

The file structure will be as follow:

```
gca
├── ...
├── data
│   ├── ...
│   ├── spbench
│   │   ├── SPBench-MV-images
│   │   │   ├── scene0025_00
│   │   │   │   ├── 1234.jpg
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── SPBench-SI-images
│   │   │   ├── scene0011_00
│   │   │   │   ├── 200.jpg
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── SPBench-MV.parquet
│   │   └── SPBench-SI.parquet
│   ├── ...
├── ...
```

## CVBench

Download the dataset from 🤗 [Hugging Face](https://huggingface.co/datasets/nyu-visionx/CV-Bench/tree/main).

The file structure will be as follow:

```
gca
├── ...
├── data
│   ├── ...
│   ├── cvbench
│   │   ├── data.json
│   │   ├── test_2d.jsonl
│   │   ├── test_2d.parquet
│   │   ├── test_3d.jsonl
│   │   ├── test_3d.parquet
│   │   └── img # After the first run of the code, it will be automatically created.
│   │       ├── 2D
│   │       └── 3D
│   ├── ...
├── ...