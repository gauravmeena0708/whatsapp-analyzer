# setup.py

from setuptools import setup, find_packages

setup(
    name="whatsapp-groupchat-analyzer",
    version="0.2.0",  # Start with an initial version
    author="Gaurav Meena",
    author_email="gauravmeena0708@gmail.com",
    description="A Python package for analyzing WhatsApp group chats.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gauravmeena0708/whatsapp-analyzer",  # Your project URL
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g.,
        # "matplotlib>=3.0",
        # "seaborn>=0.9",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose a license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version
)