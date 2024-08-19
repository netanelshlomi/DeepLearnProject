# BERT vs GPT-2: YouTube Video Titles Classifier

## Project Overview
This project compares two methods for binary classification of YouTube video titles into "Gaming and Entertainment" or "Not Gaming and Entertainment." The models used are **BERT** and **GPT-2**, fine-tuned with **LoRa** (Low-Rank Adaptation) for efficient training and reduced memory usage.

### Objective
The goal of this project is to predict whether a given YouTube video title falls under the "Gaming and Entertainment" category using two approaches:
1. Fine-tuned **BERT** model with LoRa.
2. Fine-tuned **GPT-2** model with LoRa.

## Motivation
We chose this project to experiment with well-known interactive models and apply concepts learned in our course. The classifier has potential use cases in identifying YouTube influencers, promoting gaming products, and recommending content.

## Dataset
The dataset used is the [US YouTube Trending Data](https://www.kaggle.com/datasets/datasnaek/youtube-new) from Kaggle. It contains YouTube video titles and their respective categories. After preprocessing, the dataset has ~50,000 entries, split into:
- 68% Training Set
- 12% Validation Set
- 20% Test Set

The labels are binary:
- **1**: Gaming and Entertainment
- **0**: Not Gaming and Entertainment

## Model Architectures

### GPT-2 (with LoRa)
- **Base Model**: GPT-2 LMHeadModel
- **LoRa Setup**: Fine-tuned with approximately 600k parameters (0.47% of total parameters)
- **Training Strategy**: Next-word prediction using full sentences structured as: `"Title" is a title of a video about Gaming and Entertainment`

### BERT (with LoRa)
- **Base Model**: BERTForSequenceClassification
- **LoRa Setup**: Fine-tuned with ~300k parameters (0.26% of total parameters)
- **Training Strategy**: Standard classification with input `[title, label]` pairs.

## Methods and Training
Both models were fine-tuned with the following process:
- Tokenized input fed into the model.
- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: AdamW.
- **Early Stopping**: Implemented to prevent overfitting.

For efficient fine-tuning, we only trained a fraction of the model’s total parameters using **LoRa**, which helps reduce computational requirements while maintaining performance.

## Experiments and Results
The experiments focused on evaluating model performance using **accuracy**, **precision**, **recall**, and **F1-score**. Here are the results:

| Model   | Accuracy | Precision | Recall | F1-score |
|---------|----------|-----------|--------|----------|
| GPT-2   | 79.4%    | 73.52%    | 74.27% | 73.89%   |
| BERT    | 78.75%   | 80.72%    | 85.43% | 83.00%   |
| GPT-4o  | 60.9%    | -         | -      | -        |

Observations:
- The **GPT-2 model** performed well in next-token prediction tasks, showing good capture of patterns in the "Gaming and Entertainment" category.
- **LoRa** proved to be an effective technique, reducing time and memory usage while achieving high performance.

## Future Work
There are several potential extensions for this project:
- Expanding the model to support **multi-class classification**.
- Incorporating additional **metadata** (e.g., video description or tags) to improve model accuracy.
- Exploring **larger models** like GPT-3 or more efficient models like DistilGPT-2.
- Applying **transfer learning** for tasks such as video recommendation or content categorization.
- Testing the model with **multilingual datasets** to assess its generalization to different languages.

## How to Run the Project
1. Clone the repository.
   ```bash
   git clone https://github.com/your-repo/bert-vs-gpt2-youtube-classifier.git
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/datasnaek/youtube-new) and place it in the `data/` directory.
4. Train the model.
   ```bash
   python train.py --model {bert/gpt2} --lora --epochs 5
   ```
5. Evaluate the model.
   ```bash
   python evaluate.py --model {bert/gpt2}
   ```

## License
This project is licensed under the MIT License.
