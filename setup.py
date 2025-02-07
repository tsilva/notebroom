from setuptools import setup, find_packages

setup(
    name="notebroom",
    version="0.1.0",
    py_modules=['notebroom.notebroom'],
    install_requires=[
        "nbformat>=5.9.2",
        "openai>=1.3.0",
        "python-dotenv>=1.0.0",
        "colorama>=0.4.6",
        "tqdm>=4.66.1",
    ],
    entry_points={
        'console_scripts': [
            'notebroom=notebroom.notebroom:main',
        ],
    },
)
