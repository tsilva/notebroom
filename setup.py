from setuptools import setup, find_packages

setup(
    name="notebroom",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "openai",
        "python-dotenv",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "notebroom=notebroom.main:main",
        ],
    },
    author="tsilva",
    description="A tool to improve Jupyter notebook markdown cells using AI",
    keywords="jupyter, notebook, markdown, ai",
    python_requires=">=3.6",
)
