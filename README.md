# Evaluating the Interpretability of Generative Models by Interactive Reconstruction

This repository contains code, models, and data for the CHI 2021 paper
["Evaluating the Interpretability of Generative Models by Interactive
Reconstruction."](TODO)

## Main Idea

There are now a [number](TODO) [of](TODO) [tools](TODO) for explaining and evaluating how well
humans understand _discriminative_ machine learning models. Discriminative
models generally have complicated, high-dimensional inputs (like images, audio
waveforms, blocks of text, medical timeseries of labs and vitals, etc), but
relatively low-dimensional outputs (e.g. a single 0-1 probability, or maybe a
categorical prediction over a fixed number of options). Because of this
relatively simple structure, it can often be (comparatively) easy to visualize
how discriminative model outputs change with changes in their inputs, or to
evaluate whether human users understand that relationship for different models.

However, there is another important category of machine learning models called
_generative_ models, which often have the opposite structure: their inputs can
be relatively simple (e.g. low-dimensional vectors), while their outputs are
generally complicated (e.g. the same examples as we gave for discriminative
model inputs). The purpose of generative models isn't to predict something like
a probability, but instead to model the underlying, lower-dimensional structure
in high-dimensional data. [This can be useful](TODO) when trying to understand data,
and it's also possibly to apply generative modeling techniques on a wider
variety of datasets than discriminative models (since training generative
models only requires the data, not any associated labels).

However, there is a relative dearth of research on trying to measure how well
human users can understand generative models, or even how to visualize them.
And most of the research [which](TODO) [does](TODO) [exist](TODO) focuses on
methods that project representations down to two or three dimensions, which is
often a bit too low.

In this paper, we present a method for simultaneously visualizing and
evaluating the interpretability of generative model representations. This
method is based on the idea of "interactive reconstruction," where human users
try to manipulate representation dimensions to reconstruct a target input.

The paper is available [here](TODO), and you can also try out [the actual
studies we ran](https://hreps.s3.amazonaws.com/quiz/manifest.html) live in your
web browser.

## Repository Structure

- `src/` contains Python code used for training models and computing metrics
- `data/` contains the Sinelines dataset, which we introduce, and is where the [dSprites `.npz` file](https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz) should be downloaded
- `quiz/` contains a JavaScript-based web application, built with
  [Svelte](https://svelte.dev/) and
  [TensorFlow.js](https://www.tensorflow.org/js/), that implements the
  interactive reconstruction task along with several baselines. It also
  contains the trained models we used in the study.

## Reproducing Results

### Running the Quiz

After cloning the repository, run the following commands (assuming you have [Node.js](https://nodejs.org/en/) installed):

```bash
cd quiz
npm install
npm run dev
```

This will start a local server with the quiz application on port 5000. To get
started with a list of models and tasks, visit `localhost:5000/manifest.html`.

### Re-training Models

After cloning the repository and installing [Python
dependencies](./requirements.txt), the following commands should let you
retrain all the models (except InfoGAN on MNIST; see [here](https://github.com/dtak/tensorpack/commit/929f1c819fb1943a72436d9958b2f19d96c5e6a5) for instructions):

```bash
# Retrain dSprites
dir=./quiz/public/models/dsprites
python src/train_dsprites.py --output_dir=$dir/ae  --variational=0
python src/train_dsprites.py --output_dir=$dir/vae --variational=1
python src/train_dsprites_supervised.py --output_dir=$dir/gt

# Retrain Sinelines
dir=./quiz/public/models/sinelines
python src/train_sinelines.py --output_dir=$dir/ae  --variational=0
python src/train_sinelines.py --output_dir=$dir/vae --variational=1

# Retrain MNIST
dir=./quiz/public/models/mnist
python src/train_mnist.py --output_dir=$dir/ae5  --K=5  --variational=0
python src/train_mnist.py --output_dir=$dir/vae5 --K=5  --variational=1 --kl_penalty=1
python src/train_mnist.py --output_dir=$dir/tc5  --K=5  --variational=1 --kl_penalty=1 --tc_penalty=9
python src/train_mnist.py --output_dir=$dir/ss5  --K=5  --variational=1 --kl_penalty=1 --tc_penalty=9 --semi_supervised=1
python src/train_mnist.py --output_dir=$dir/ae10 --K=10 --variational=0
python src/train_mnist.py --output_dir=$dir/tc10 --K=10 --variational=1 --kl_penalty=1 --tc_penalty=9
python src/train_mnist.py --output_dir=$dir/ss10 --K=10 --variational=1 --kl_penalty=1 --tc_penalty=9 --semi_supervised=1
```

Note that the code has been lightly refactored since the original study for
clarity (and models have been given more human-friendly names), but it should
behave identically.

Note also that you may face TensorFlow.js conversion compatibility
issues if using versions other than those listed in the [requirements
file](./requirements.txt), as non-backwards-compatible changes have been made
to this process since this project's inception.

## Citation

TODO
