# Contributing Guidelines

Firstly a great welcome for anyone who wants to participate in TinyMS community👏. For those who are not familiar with how this community works, these are some guidelines to get you quickly getting started.

## Contributor License Agreement

It's required to sign CLA before your first code submission to TinyMS community.

For individual contributor, please refer to [ICLA sign page](https://cla-assistant.io/tinyms-ai/tinyms) for the detailed information.

## Getting Started

- Fork the repository on [GitHub](https://github.com/tinyms-ai/tinyms).
- Read the [README.md](https://github.com/tinyms-ai/tinyms/blob/main/README.md) and [install page](https://tinyms.readthedocs.io/en/latest/quickstart/install.html) for project information and build instructions.
- Try our first [quick start tutorial](https://tinyms.readthedocs.io/en/latest/quickstart/quickstart_in_one_minute.html) in one minutes!😍

## Contribution Workflow

### Code style

Please follow this style to make TinyMS easy to review, maintain and develop.

* Coding guidelines

    The *Python* coding style suggested by [Python PEP 8 Coding Style](https://pep8.org/) is adopted in TinyMS community.

* Unittest guidelines

    The *Python* unittest style suggested by [pytest](http://www.pytest.org/en/latest/) is adopted in TinyMS community.

* Autodoc guidelines

    The *Autodoc* generated style suggested by [Sphinx](https://www.sphinx-doc.org/en/master/) is adopted in TinyMS community.

### Fork-Pull development model

* Fork TinyMS repository

    Before submitting code to TinyMS project, please make sure that this project have been forked to your own repository. It means that there will be parallel development between TinyMS repository and your own repository, so be careful to avoid the inconsistency between them.

    > **NOTICE:** The default branch name of TinyMS project is `main` instead of `master`.

* Clone the remote repository

    If you want to download the code to the local machine, `git` is the best choice:

    ```shell
    git clone https://github.com/{insert_your_forked_repo}/tinyms.git
    git remote add upstream https://github.com/tinyms-ai/tinyms.git
    ```

* Develop code locally

    To avoid inconsistency between multiple branches, checking out to a new branch for every pull request is `SUGGESTED`:

    ```shell
    git checkout -b {new_branch_name}
    ```

    > **NOTICE:** Please try to pull the latest code from upstream repository (`git pull upstream main`) every time before checking out a new branch.

    Then you can change the code arbitrarily.

* Push the code to the remote repository

    After updating the code, you should push the update in the formal way:

    ```shell
    git add .
    git status # Check the update status
    git commit -m "Your commit title"
    git commit -s --amend # Add the concrete description of your commit
    git push origin {new_branch_name}
    ```

* Pull a request to TinyMS repository

    In the last step, your need to pull a compare request between your new branch and TinyMS `main` branch. After finishing the pull request, the Travis CI will be automatically set up for building test.

### Report issues

A great way to contribute to the project is to send a detailed report when you encounter an issue. We always appreciate a well-written, thorough bug report, and will thank you for it!🤝

When reporting issues, refer to this format:

- What version of env (tinyms, mindspore, os, python etc) are you using?
- Is this a BUG REPORT or FEATURE REQUEST?
- What happened?
- What you expected to happen?
- How to reproduce it? (as minimally and precisely as possible)
- Special notes for your reviewers?

**Issues advisory:**

- **If you find an unclosed issue, which is exactly what you are going to solve,** please put some comments on that issue to tell others you would be in charge of it.
- **If an issue is opened for a while,** it's recommended for contributors to precheck before working on solving that issue.
- **If you resolve an issue which is reported by yourself,** it's also required to let others know before closing that issue.

### Propose PRs

**Working on your first Pull Request?** 📚You can learn how from this *free* series [How to Contribute to an Open Source Project on GitHub](https://kcd.im/pull-request)📚

When proposing pull requests, please adhere to these rules:

- Raise your idea as an *issue* on [GitHub](https://github.com/tinyms-ai/tinyms/issues).
- If it is a new feature that needs lots of design details, a design proposal should also be submitted.
- After reaching consensus in the issue discussions and design proposal reviews, complete the development on the forked repo and submit a PR.
- None of PRs is not permitted until it receives **2+ LGTM** from approvers. Please NOTICE that approver is NOT allowed to add *LGTM* on his own PR.
- After PR is sufficiently discussed, it will get merged, abandoned or rejected depending on the outcome of the discussion.

**PRs advisory:**

- Any irrelevant changes should be avoided.
- Make sure your commit history being ordered.
- Always keep your branch up with the master branch.
- For bug-fix PRs, make sure all related issues being linked.

## Community Support

Whenever you feel confused in this community, please be free to reach out the help from TinyMS community with these approaches:

- **Wechat communication.** Add `mindspore0328` Wechat ID to ask for help.
- **QQ group.** TBD.
- **Slack channel.** Join `tinyms` channel in [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) to communicate with each other.
