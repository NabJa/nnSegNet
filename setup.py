import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep-brats",
    version="0.0.1",
    author="Nabil Jabareen",
    author_email="nabil.jabareen@charite.de",
    description="Deep Learning for the Multimodal Brain Tumor Segmentation (BraTS) Challange.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/lukassen/glioblastoma/brats21",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires="==3.8.10",
    entry_points={
        "console_scripts": [
            "brats21_optimize_thresholds = brats21.optmization:main",
            "brats21_postprocess = brats21.postprocessing:main",
            "brats21_evaluation = brats21.evaluation:main",
        ]
    },
)
