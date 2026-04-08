# Headlines-Bert-T5-GPT2

This repository contains the code and resources for the Rutgers-Bert-T5-GPT2 project, which focuses on natural language processing (NLP) tasks using a combination of BERT, T5, and GPT-2 models. The project aims to leverage the strengths of these models to improve performance on various NLP tasks such as text classification, summarization, and language generation.


## Table of Contents

- [Rutgers-Bert-T5-GPT2](#rutgers-bert-t5-gpt2)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)

## Installation

To install the necessary dependencies for this project, please follow the instructions below:

1. Clone the repository:

   ```bash
   git clone rutgers-bert-t5-gpt2.git
   cd rutgers-bert-t5-gpt2
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the models for various NLP tasks, you can run the provided scripts in the `src` directory. For example, to perform text classification, you can run:

```bash
python src/text_classification.py --input data/input.txt --output results/classification_output.txt
```
Make sure to replace the input and output file paths with your desired locations. You can also explore other scripts for summarization and language generation tasks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more.


