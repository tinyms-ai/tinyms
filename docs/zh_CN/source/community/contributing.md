# 贡献指导

首先非常欢迎加入TinyMS社区的各位开发者👏，如果您还不太清楚TinyMS社区的运作流程，可通过本篇指导文档的学习快速上手社区贡献。

## 开源社区贡献者协议

在您首次向TinyMS社区提交代码的时候，请知悉在代码合入之前需签署CLA（Contributor License Agreement）协议。

针对个人开发者，请访问[CLA签署页面](https://cla-assistant.io/tinyms-ai/tinyms)进行协议签署。

## 快速上手TinyMS

- Fork TinyMS社区[GitHub代码仓](https://github.com/tinyms-ai/tinyms)
- 认真阅读TinyMS[项目简介](https://github.com/tinyms-ai/tinyms/blob/main/README.md)和[安装页面](https://tinyms.readthedocs.io/en/latest/quickstart/install.html)
- 尝试我们的[一分钟上手教程](https://tinyms.readthedocs.io/en/latest/quickstart/quickstart_in_one_minute.html)😍

## 贡献流程

### 代码风格

当前TinyMS社区制定了编码风格、单元测试以及文档自动生成等方面的规范，请在提交代码之前仔细阅读社区开发规范。

- 编码风格：Python语言为主，基于[Python PEP 8 Coding Style](https://pep8.org/)进行开发
- 单元测试：Python语言为主，基于[pytest](http://www.pytest.org/en/latest/)进行开发
- 自动文档生成：基于[Sphinx](https://www.sphinx-doc.org/en/master/)进行社区文档的自动生成

### Fork-Pull开发模式

* Fork TinyMS代码仓

    在提交代码之前，请您务必确认TinyMS项目已Fork至您的个人仓。同时，这将意味着在TinyMS官方仓和您的Fork仓之间会存在并行开发的情况，甚至可能造成分支冲突的问题，因此请您在个人仓开发过程中格外注意与主仓代码不一致的情况。

    > **注意：** 当前TinyMS项目的默认分支名已设置为`main`。

* 克隆远端代码

    如果您想把代码克隆到本地进行开发，那么建议您使用`git`工具：

    ```shell
    git clone https://github.com/{insert_your_forked_repo}/tinyms.git
    git remote add upstream https://github.com/tinyms-ai/tinyms.git
    ```

* 本地开发调试

    考虑到多个分支的同步问题，建议您针对每个Pull Request新建对应的分支名：

    ```shell
    git checkout -b {new_branch_name}
    ```

    > **注意：** 建议您每次在切换分支之前都要运行`git pull upstream main`指令拉取最新代码。

* 推送代码

    本地开发调试之后，您可以通过如下指令生成git commit记录：

    ```shell
    git add .
    git status # Check the update status
    git commit -m "Your commit title"
    git commit -s --amend # Add the concrete description of your commit
    git push origin {new_branch_name}
    ```

* 提交PR

    最后您需要在网页创建一个Pull Request，源分支和目标分支分别设置为您个人仓的新建分支和TinyMS主仓的`main`分支。提交PR之后，系统会自动触发Travis CI工作来确认您的修改是否符合要求。

### 上报漏洞

为项目做贡献的一种好方法是在遇到问题时发送详细的报告。我们始终感谢写得好的、详尽的错误报告，希望您能遵守社区规则！🤝

报告问题时，请参考以下格式：

- 您使用的是哪个版本的环境信息（tinyms、mindspore、os、python等）？
- 这是错误报告还是特性需求？
- 发生了什么？
- 您预期会发生什么？
- 如何复现它（尽可能少且精确）？
- 给代码审核人员的特别提示？

**漏洞咨询：**

- **如果您发现未解决的问题，而这正是您要解决的问题**，请对该问题发表一些评论，以告诉其他人您将负责此事。
- **如果某个问题开放了一段时间**，建议贡献者在解决该问题之前进行预先检查。
- **如果您解决了自己上报的问题**，还需要在解决该问题之前让他人知道。

### 提交更改

- 在[GitHub](https://github.com/tinyms-ai/tinyms/issues)上将您的想法作为*issue*提出。
- 如果这是一项需要大量设计细节的新功能，则还应提交设计建议。
- 在问题讨论和设计方案审查中达成共识后，完成分叉存储库的开发并提交PR。
- 除非获得批准者的**2+ LGTM**，否则不允许任何PR。请注意不允许批准者在自己的PR上添加 *LGTM*的行为。
- 对PR进行充分讨论之后，它将根据讨论的结果而被合并、放弃或拒绝。

**PR咨询：**

- 应避免任何不相关的更改。
- 确保您的提交历史记录逻辑清晰。
- 始终使您的提交分支与主分支保持一致。
- 对于错误修复PR，请确保所有相关问题都已链接。

## 社区支持

如果您在使用TinyMS过程中遇到任何问题，请立马通过如下渠道向社区寻求帮助：

- **微信**。请搜索`mindspore0328`微信号添加好友
- **QQ群**。TBD
- **Slack**。请打开[MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w)加入`tinyms`小组与他人进行交流
