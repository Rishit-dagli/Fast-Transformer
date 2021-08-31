# Fast Transformer [![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2FRishit-dagli%2FFast-Transformer)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FRishit-dagli%2FFast-Transformer)

![PyPI](https://img.shields.io/pypi/v/fast-transformer)
[![Lint Code Base](https://github.com/Rishit-dagli/Fast-Transformer/actions/workflows/linter.yml/badge.svg)](https://github.com/Rishit-dagli/Fast-Transformer/actions/workflows/linter.yml)
[![Upload Python Package](https://github.com/Rishit-dagli/Fast-Transformer/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Rishit-dagli/Fast-Transformer/actions/workflows/python-publish.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![GitHub License](https://img.shields.io/github/license/Rishit-dagli/Fast-Transformer)
[![GitHub stars](https://img.shields.io/github/stars/Rishit-dagli/Fast-Transformer?style=social)](https://github.com/Rishit-dagli/Fast-Transformer/stargazers)
[![GitHub followers](https://img.shields.io/github/followers/Rishit-dagli?label=Follow&style=social)](https://github.com/Rishit-dagli)
[![Twitter Follow](https://img.shields.io/twitter/follow/rishit_dagli?style=social)](https://twitter.com/intent/follow?screen_name=rishit_dagli)

This repo implements [Fastformer: Additive Attention Can Be All You Need](https://arxiv.org/abs/2108.09084) by Wu et al. in 
TensorFlow. **Fast Transformer** is a Transformer variant based on additive attention that can handle long sequences 
efficiently with linear complexity. Fastformer is much more efficient than many existing Transformer models and can 
meanwhile achieve comparable or even better long text modeling performance.

![Architecture](media/architecture.png)

## Installation
Run the following to install:

```sh
pip install fast-transformer
```

## Developing fast-transformer
To install `fast-transformer`, along with tools you need to develop and test, run the following in your virtualenv:

```sh
git clone https://github.com/Rishit-dagli/Fast-Transformer.git
# or clone your own fork

cd fast-transformer
pip install -e .[dev]
```

## Want to Contribute 🙋‍♂️?

Awesome! If you want to contribute to this project, you're always welcome! See [Contributing Guidelines](CONTRIBUTING.md). You can also take a look at [open issues](https://github.com/Rishit-dagli/Fast-Transformer/issues) for getting more information about current or upcoming tasks.

## Want to discuss? 💬

Have any questions, doubts or want to present your opinions, views? You're always welcome. You can [start discussions](https://github.com/Rishit-dagli/Fast-Transformer/discussions).