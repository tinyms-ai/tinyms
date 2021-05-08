# What is TinyMS

## TinyMS Project

### Introduction

* TinyMS is an open source deep learning development kit written in Python. It runs on top of deep learning framework like [MindSpore](https://github.com/mindspore-ai/), and provides high level API covers the entire lifecycle and workflow for AI development that ranges from data preparation to model deployment.
* TinyMS is composed of several modules including *data*, *model* and *serving*. It provides transform data processing operators for different scenarios and reuses MindSpore datasets like cifar-10.
* TinyMS are mainly designed to help user groups like deep learning beginners, researchers who conducts studies related to deep learning, and AI application developers with a crash course.
* Combined with video tutorials (available in Chinese on [Bilibili](https://www.bilibili.com/video/BV1MB4y1P79S) site and soon with English version on YouTube channel), TinyMS project offers the best deep learning crash course and entry level development experience todate.

### TinyMS vs Keras

To quote from [Keras' website](https://keras.io/about/)
> Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow.
> It was developed with a focus on enabling fast experimentation.
> Being able to go from idea to result as fast as possible is key to doing good research.

Keras is known for its completeness. Keras is composed of modules such as *dataset*, *layer*, *model* and *backend*. It provides commonly used datasets and data processing functions for different scenarios. The *layer* module provides all encompassing functionalities such as convolution, embedding, pooling, backend (multiple support for Tensorflow, CNTK and Theano). The *model* module provides functionalities such as model types (sequential or functional), network construction (input, output, pooling, etc.), model compilation, model training, model verification and inference.

In comparison, TinyMS designs a set of more abstract high level APIs and therefore is less complex than Keras. For example one can complete dataset preprocessing with just one line of code in TinyMS. TinyMS also provides several individual tools and quick model deployment module which Keras has not yet offered.

### TinyMS vs Fastai

To quote from fastai's [README](https://github.com/fastai/fastai#about-fastai)
> fastai is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches.

With the strength of PyTorch's flexibility, Fastai could provide out-of-the-box development experience for model types like *vision*, *text*, *tabular*, *collab*. However unlike Keras's multi-backend support, Fastai's backend is tightly coupled with PyTorch and its versioning.

Fastai is known for its "petitness" which provides a great lightweight and easy-to-understand structure. Fastai consists of three major modules: *data*, *models* and *learner*. The *data* module provide transform data preprocessing operations which is convenient for developers. The *model* module provides many predefined networks like unet for quick model construction. The *learner* module defines the relationship between data and model with a set of callback functions to help developers quickly grasp the most common deep learning architectures. It provides a rich set of tools such as data downloading, decompression, figure verification and file processing.

In comparison, while sharing similar design concepts on high level APIs, TinyMS offers predefined MindSpore datasets which could help developers with dataset processing enormously as well as quick model deployment module, both of which Fastai has not yet provided.

## TinyMS Community

Other than TinyMS project，the community at-large also includes the many other projects and activities：

* Specification project：an attempt to standardize the format of model training scripts. Due to the abstract nature of TinyMS APIs, we found it necessary to have a standard or guideline for ModelZoo.
* <https://tinyms-ai.github.io>：a simple website built in open source based upon GitHub Page mechanism.
* RustedAI Team：only visible for org members at the moment, RustedAI is an initiative that TinyMS tries to build for more adoption of Rust-lang in the field of deep learning to meet the goal of low runtime footprint.
* Community Activities：We will organize TinyMS model competitions and many other activities including Meetups, webinars, etc.

## TinyMS and Developers

As a new open source project, TinyMS stands on the shoulder of the giants like Keras and Fastai. Although we hope to achieve many innovations in our design, it still depends on a vibrant community and ecosystem to make TinyMS reach the depth and broadness of its predecessors in order to better serve the academia, industry and developers in general.
