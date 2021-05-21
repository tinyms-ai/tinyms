# 什么是TinyMS

## TinyMS项目

### TinyMS简介

* TinyMS是一款主要用PyThon语言编写的开源深度学习开发工具包，基于以MindSpore为代表的新型开源深度学习框架，提供面向从数据准备到模型部署全流程的极简易用的高阶API封装，并通过易于扩展的模块化设计，提供覆盖多种业务场景的能力。
* 主要由data, model, serving等模块组成，分场景分领域提供transform数据预处理算子，复用MindSpore原生数据集提供常用数据集，如：cifar-10等。data提供部分自定义数据集和常用的数据下载和解压等常用工具集，model提供常用的预置模型，并提供模型构建，模型编译，模型训练、验证与推理。serving通过搭建服务器来提供AI模型应用服务,为新手提供快速推理的体验。
* TinyMS面向的主要用户群体为深度学习初学者、研究领域涉及深度学习结合的科研人员、以及深度学习相关业务应用开发的企业开发人员。
* 通过搭配完整的在线课程教学，TinyMS提供目前业界最佳的深度学习入门与开发体验。

### TinyMS vs Keras

Keras 是一个用 Python 编写的高级神经网络 API，将把用户体验放在首要位置，支持短时间内出实验。

Keras项目可以说是“大而全”，主要由dataset, layer, model和backend模块构成，提供较多常用的预置数据集，并分场景分领域提供数据预处理函数，layer网络层提供较完善，如：convolution卷积层，embedding嵌入层，pooling池化层等。backend支持多个后端（TensorFlow、CNTK和Theano），与TensorFlow版本不强耦合。Model提供模型选择（sequential）、网络层构建（输入层、输出层和池化层等），模型编译，模型训练、验证与推理。

TinyMS在高阶API方面会更为简单抽象，较Keras来说复杂度更低，比如提供了只需一行代码即可完成数据集的预处理，而且在设计中重点考虑到了Keras尚未提供单独好用的工具库，以及尚未提供的快速部署推理模块等。

### TinyMS vs Fastai

Fastai是为了帮助新手快速轻松出结果的高阶API项目, 其基于PyTorch的深度学习库，利用底层PyTorch库的灵活性，分领域分场景地提供包括对vision，text，tabular和collab模型的“开箱即用”的支持，后端对PyTorch的版本要求紧耦合。

fastai 深度学习库项目较轻便，目录清晰易理解，可以说是“小而美”，主要由data, models和learner三大模块构成，其中，data提供了transform类方便开发者进行数据预处理操作。models按应用领域，提供部分预置网络，如：unet，快速实现模型构建。learner实现数据和模型的关联，并定义了一系列回调函数，帮助开发者快速厘清深度学习的架构，提供模型训练、模型评估、模型保存与加载和模型推理。除此之外，还拥有较丰富好用的工具集，如：数据下载，解压，图片验证和文件处理等工具函数。

TinyMS在高阶API方面理念与Fastai相近，不同点在于TinyMS提供了常用的MindSpore预置数据集，方便开发者简化对数据集的调用，而且提供了Fastai尚未提供的快速部署推理模块等。

## TinyMS开源社区

TinyMS开源社区中除了TinyMS项目外，还有如下一些项目和活动：
* Specification项目：主要用来协作制定面向模型训练脚本的格式规范。由于TinyMS提供了较为高阶的API抽象，因此诞生了ModelZoo脚本规范性和标准化的需求，便于高阶封装的持续迭代
* <https://tinyms-ai.github.io>：开源实现的简单官方网站搭建，基于Github Page
* RustedAI Team：目前只有组织成员可见，RustedAI是TinyMS旨在推动利用Rust语言编写更多的低运行时开销的深度学习组件
* 社区活动：我们会不定期的组织TinyMS模型拉力赛，以及多种多样的Meetup活动

## TinyMS与开发者

TinyMS是一个新生的开源项目，我们站在Keras、Fastai等巨人的肩膀上，虽然在设计理念上有所创新，但依然需要社区开发者一起持续的协作，才能达到可以更好的服务学术界、产业界和开发者的深度和广度。
