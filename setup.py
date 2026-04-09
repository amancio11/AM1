from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="ai-glass-cleanliness",
    version="1.0.0",
    author="AI Challenge Team",
    description="AI system for drone-based building facade glass cleanliness detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-glass=training.train_glass:main",
            "train-dirt=training.train_dirt:main",
            "train-multitask=training.train_multitask:main",
            "infer-image=inference.image_inference:main",
            "infer-video=inference.video_inference:main",
            "evaluate=scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
