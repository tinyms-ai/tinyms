#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""setup package."""
import os
import setuptools

package_name = 'tinyms'
version_tag = '0.2.1'
pwd = os.path.dirname(os.path.realpath(__file__))


def _read_file(filename):
    with open(os.path.join(pwd, filename), encoding='UTF-8') as f:
        return f.read()


readme = _read_file('README.md')


def _write_version(file):
    file.write("__version__ = '{}'\n".format(version_tag))


setup_required_package = [
    'wheel >= 0.32.0',
    'setuptools >= 40.8.0',
]


install_required_package = [
    'numpy >= 1.17.0',
    'easydict >= 1.9',
    'scipy >= 1.5.2',
    'matplotlib >= 3.1.3',
    'Pillow >= 6.2.0',
    'mindspore == 1.2.0',
    'requests >= 2.22.0',
    'flask >= 1.1.1',
    'python-Levenshtein >= 0.10.2',
    'gensim >= 3.8.1',
    'PyYAML',
]

test_required_package = [
    'pycocotools >= 2.0.0',
]

package_data = {
    'tinyms.hub': [
        'assets/tinyms/*/*.yaml',
    ]
}

if __name__ == "__main__":
    with open(os.path.join(pwd, package_name, 'version.py'), 'w') as f:
        _write_version(f)

    setuptools.setup(
        name=package_name,
        version=version_tag,
        author='The TinyMS Authors',
        author_email='wanghui71leon@gmail.com',
        url='https://tinyms.readthedocs.io/en/latest/',
        download_url='https://github.com/tinyms-ai/tinyms/tags',
        project_urls={
            'Sources': 'https://github.com/tinyms-ai/tinyms',
            'Issue Tracker': 'https://github.com/tinyms-ai/tinyms/issues',
        },
        description='TinyMS is an Easy-to-Use deep learning development toolkit.',
        long_description="\n\n".join([readme]),
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        package_data=package_data,
        include_package_data=True,
        python_requires='>=3.7',
        setup_requires=setup_required_package,
        install_requires=install_required_package,
        tests_require=test_required_package,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords='machine learning toolkit',
    )
