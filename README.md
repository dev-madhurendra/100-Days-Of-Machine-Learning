## Table of Contents
1. [What is Machine Learning?](#what-is-machine-learning)
2. [AI vs ML vs DL](#ai-vs-ml-vs-dl)

## What is Machine Learning?

Machine learning is a subset of artificial intelligence (AI) that focuses on teaching computers to learn from data and make predictions or decisions without being explicitly programmed to do so. It enables computers to learn patterns from data and utilize that knowledge to perform tasks or make predictions.

### Explanation with Example

Imagine you want to build a program that can predict whether an email is spam or not spam (ham). Instead of manually writing rules to classify emails, you can use machine learning.

1. **Data Collection:** Gather a dataset of emails, each labeled as spam or not spam (ham).
   
2. **Feature Extraction:** Extract features from the emails, such as word frequency, presence of specific phrases, or email length.

3. **Training Phase:** Feed the labeled dataset into a machine learning algorithm. The algorithm learns from the data, identifying patterns that differentiate spam from non-spam emails.

4. **Model Building:** After training, the algorithm builds a model representing learned patterns. The model captures the relationship between email features and their labels (spam or not spam).

5. **Prediction:** When a new email arrives, input its features into the trained model. The model analyzes these features and predicts whether the email is spam or not based on learned patterns.

6. **Evaluation and Improvement:** Evaluate the model's performance using a separate test dataset. Refine the model by adjusting parameters, using different algorithms, or gathering more data.

## AI vs ML vs DL
![AI, ML, DL](https://siddhithakkar.com/wp-content/uploads/2020/02/ai-ml-dl-1-e1582524532998.jpg)

| Aspect          | Artificial Intelligence (AI) | Machine Learning (ML) | Deep Learning (DL) |
|-----------------|------------------------------|-----------------------|--------------------|
| Definition      | AI is a broad field of computer science that aims to create machines or systems that can perform tasks that typically require human intelligence. | ML is a subset of AI that focuses on teaching computers to learn from data and make predictions or decisions without being explicitly programmed to do so. | DL is a subset of ML that uses neural networks with many layers (deep architectures) to learn from large amounts of data. |
| Scope           | AI covers a wide range of techniques and applications, including problem-solving, natural language processing, robotics, computer vision, and more. | ML focuses specifically on algorithms and models that learn from data to perform tasks or make predictions. | DL specializes in training neural networks with multiple layers to automatically discover patterns and features from data. |
| Approach        | In AI, various techniques are employed, including rule-based systems, expert systems, and statistical methods, to simulate human intelligence. | ML algorithms learn from data by identifying patterns and making predictions or decisions based on those patterns. | DL algorithms use deep neural networks, which consist of many layers of interconnected nodes, to automatically learn hierarchical representations of data. |
| Data Dependency| AI systems may or may not rely heavily on large amounts of data, depending on the specific application and approach used. | ML algorithms require labeled or unlabeled data for training to learn patterns and make predictions. The performance often improves with more high-quality data. | DL algorithms typically require large datasets for training, as they need vast amounts of data to effectively learn complex features and patterns. |
| Examples        | Examples of AI include virtual assistants (like Siri or Alexa), self-driving cars, recommendation systems, and facial recognition systems. | Examples of ML applications include spam email detection, image recognition, language translation, and recommendation systems (like those used by Netflix or Amazon). | Examples of DL applications include image and speech recognition, natural language processing, autonomous vehicles, and medical diagnosis systems. |

In summary:
- AI is the broader field focused on creating intelligent systems.
- ML is a subset of AI that teaches computers to learn from data and make predictions or decisions.
- DL is a subset of ML that uses deep neural networks to learn from large amounts of data.
