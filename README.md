## Overview of the Prototype

This prototype is designed to evaluate the reliability of **large language model (LLM) outputs** by verifying **logical consistency** and **factual accuracy**. It consists of three main components:
1. Paraphrase-Based Logical Consistency Checking  
2. Fact Verification Using Wikipedia  
3. Evaluation and Output  

This prototype enables automated verification of LLM-generated content, serving as a useful tool for diagnosing inconsistencies and misinformation.

## Installation and Usage

This prototype was developed and tested in **Google Colab** for ease of use. However, if you want to run it locally in **Jupyter Notebook**, follow the steps below.

### Running in Google Colab  
Simply upload the script to **Google Colab**, ensure that the runtime is set to **GPU**, and run the cells sequentially.

### Running Locally in Jupyter Notebook  
To run the script on your local machine using **Jupyter Notebook**, follow these steps:

- Install the required dependencies: transformers, sentence-transformers, torch, accelerate, wikipedia

- Launch Jupyter Notebook
