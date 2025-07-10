from setuptools import setup, find_packages

setup(
    name='IHSetDean91',
    version='2.1.2',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pathlib',
        'matplotlib'
    ],
    author='Mario Luiz Mascagni',
    author_email='tatramario@gmail.com',
    description='IH-SET Dean (1991)',
    url='https://github.com/IHCantabria/IHSetDean',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
		'Programming Language :: Python :: 3.11',
    ],
)

