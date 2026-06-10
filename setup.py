from setuptools import setup, find_packages

setup(
    name="daphnia-tracking",
    version="1.0.0",
    description="Автоматизированная система биоиндикации качества водных сред на основе анализа поведения Daphnia spp.",
    author="Порошин Алексей Васильевич",
    author_email="lexalm259@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "opencv-python>=4.5.0",
        "ultralytics>=8.0.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    entry_points={
        "console_scripts": [
            "daphnia-track=scripts.validate_system:main",
        ],
    },
)