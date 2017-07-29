import io

from distutils.util import convert_path
from setuptools import setup, find_packages

main_ns = {}
ver_path = convert_path('jagular/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.rst')

setup(
    name='jagular',
    version=main_ns['__version__'],
    url='https://github.com/kemerelab/jagular/',
    download_url = 'https://github.com/kemerelab/jagular/tarball/' + main_ns['__version__'],
    license='MIT License',
    author='Joshua Chu, Etienne Ackermann',
    install_requires=['numpy>=1.9.0'
                    ],
    author_email='jpc6@rice.edu',
    description='Out-of-core pre-processing of big-ish electrophysiology data.',
    long_description=long_description,
    packages=find_packages(),
    keywords = "electrophysiology neuroscience data analysis",
    include_package_data=True,
    platforms='any'
)
