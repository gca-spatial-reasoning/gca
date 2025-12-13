<p align="center">

<h1 align="center"><strong>Geometrically-Constrained Agent for Spatial Reasoning</strong></h1>
</div>

<p align="center">
    <a href="https://github.com/Zx55">Zeren Chen</a><sup>1,2*</sup>,</span>
    <a href="https://github.com/ursulalujun">Xiaoya Lu</a><sup>2,3*</sup>,</span>
    <a href="#">Zhijie Zheng</a><sup>1,2</sup>,
    <a href="#">Pengrui Li</a><sup>1</sup>,
    <a href="#">Lehan He</a><sup>1,4</sup>,
    <a href="#">Yijin Zhou</a><sup>2,3,4</sup>,
    <a href="https://amandajshao.github.io">Jing Shao</a><sup>2</sup>,
    <a href="#">Bohan Zhuang</a><sup>5†</sup>,
    <a href="#">Lu Sheng</a><sup>1†</sup>
</p>

<p align="center">
    <sup>1</sup>School of Software, Beihang University,
    <sup>2</sup>Shanghai AI Laboratory,
    <sup>3</sup>Shanghai Jiao Tong University,
    <sup>4</sup>Shanghai Innovation Institute,
    <sup>5</sup>ZIP Lab, Zhejiang University
</p>

<p align="center">
    <sup>*</sup>Equal Contribution &nbsp;&nbsp;
    <sup>&dagger;</sup>Corresponding Author
</p>

<p align="center">
    <a href="https://arxiv.org/pdf/2511.22659">📑 Paper</a>  |
    <a href="https://arxiv.org/abs/2511.22659">📖 arXiv</a>  |
    <a href="https://gca-spatial-reasoning.github.io">🌐 Homepage</a>
</p>

## 🏠 About
<div style="text-align: center;">
    <img src="assets/docs/teaser.jpg" alt="Dialogue_Teaser" width=100% >
</div>
    <strong>Geometrically-Constrained Agent (GCA)</strong> resolves the semantic-to-geometric gap by decoupling the reasoning process into Task Formalization and Constrained Geometric Computation.

## 📢 News
- [Coming Soon!] 📝 We will release our code.
- [2025-12-1] 🔥 We release the [paper](https://arxiv.org/pdf/2511.22659) of GCA.

## ⚙️ Quick Start

### Installation

The code requires `python>=3.11` and `torch>=2.5.1`. Please follow the instructions [here](docs/install.md) to install the dependencies and third party repositories.

### Eval Dataset

We evaluate GCA on several challenging spatial reasoning benchmarks, including MMSI-Bench, MindCube, OmniSpatial, SPBench and CVBench. Please follow the instructions [here](docs/dataset.md) to prepare these evaluation datasets.

### Usage

For detailed configuration (JSON/Env Vars/CLI) and VLM deployment instructions, please refer to the [Usage Documentation](docs/usage.md).

Run the GCA on supported benchmarks (MMSI, MindCube, CVBench, etc.):

```bash
python -m entrypoints.agent --benchmark mmsi --concurrency 16
```

## 🔗 Citation

If you find our work and this codebase helpful, please consider starring this repo 🌟 and cite:

```bibtex
@article{chen2025geometrically,
      title={Geometrically-Constrained Agent for Spatial Reasoning}, 
      author={Zeren, Chen and Xiaoya, Lu and Zhijie, Zheng and Pengrui, Li and Lehan, He and Yijin, Zhou and Jing, Shao and Bohan, Zhuang and Lu, Sheng},
      journal={arXiv preprint arXiv:2511.22659},
      year={2025}
}
```
