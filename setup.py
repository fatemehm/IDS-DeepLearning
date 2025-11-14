from setuptools import setup, find_packages

setup(
    name="iot-ids-mlops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "fastapi>=0.104.0",
        "mlflow>=2.8.0",
    ],
    author="Fatemeh M",
    author_email="fatemehm@github.com",
    description="ML-based Intrusion Detection System for IoT devices",
    url="https://github.com/fatemehm/IDS-DeepLearning",
    python_requires=">=3.12",
)
