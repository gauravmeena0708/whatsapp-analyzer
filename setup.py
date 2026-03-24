# setup.py

from setuptools import setup, find_packages

setup(
    name='whatsapp-groupchat-analyzer',
    version='1.0.7',  # Increment the version number
    author="Gaurav Meena",
    author_email="gauravmeena0708@gmail.com",
    description="A Python package for analyzing WhatsApp group chats.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gauravmeena0708/whatsapp-analyzer",  # Your project URL
    packages=find_packages(),
    install_requires=[
        "beautifulsoup4",
        "seaborn>=0.9",
        "textblob",
        "emoji",
        "networkx",
        "matplotlib",
        "wordcloud",
        "nltk",
        "scikit-learn",
        "python-dateutil",
        "pandas",
        "numpy",
        "regex",
        "pdfkit",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose a license
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "whatsapp-analyzer=whatsapp_analyzer.run:main",
        ],
    },
    python_requires=">=3.7",  # Minimum Python version
)
