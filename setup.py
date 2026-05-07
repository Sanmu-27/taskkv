from setuptools import setup, find_packages

setup(
    name="taskkv",
    version="0.1.0",
    description="Task-Aware KV Cache Compression for LLM Agents",
    author="Anonymous",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "serving": ["vllm>=0.3.0"],
        "eval": ["alfworld", "agentbench"],
    },
)
