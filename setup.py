from setuptools import setup, find_packages
import codecs

# Читаем README.md с правильной кодировкой
with codecs.open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="magic_classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Классификатор магических трюков",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/magic_classifier",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 