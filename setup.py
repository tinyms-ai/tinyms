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
version_tag = '0.1.0'
pwd = os.path.dirname(os.path.realpath(__file__))

def _read_file(filename):
    with open(os.path.join(pwd, filename), encoding='UTF-8') as f:
        return f.read()


readme = _read_file('README.md')
release = _read_file('RELEASE.md')


def _write_version(file):
    file.write("__version__ = '{}'\n".format(version_tag))


required_package = [
    'scipy >= 1.5.3',
    'mindspore >= 1.1.0',
    'wheel >= 0.32.0',
    'setuptools >= 40.8.0',
]

if __name__ == "__main__":
    with open(os.path.join(pwd, package_name, 'version.py'), 'w') as f:
        _write_version(f)

    setuptools.setup(
        name=package_name,
        version=version_tag,
        author='The TinyMS Authors',
        author_email='wanghui71leon@gmail.com',
        url='https://github.com/tinyms-ai/tinyms',
        download_url='https://github.com/tinyms-ai/tinyms/tags',
        project_urls={
            'Sources': 'https://github.com/tinyms-ai/tinyms',
            'Issue Tracker': 'https://github.com/tinyms-ai/tinyms/issues',
        },
        description='TinyMS is an Easy-to-Use deep learning development toolkit.',
        long_description="\n\n".join([readme, release]),
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        python_requires='>=3.7',
        install_requires=required_package,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: C++',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords='machine learning toolkit',
    )
