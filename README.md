## Table of Contents
1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [AI vs ML vs DL](#2-ai-vs-ml-vs-dl)
3. [Types of Machine Learning](#3-types-of-machine-learning)_
4. [Key differences between types of machine learning](#4-key-differences-between-types-of-machine-learning)

## 1. What is Machine Learning?

Machine learning is a subset of artificial intelligence (AI) that focuses on teaching computers to learn from data and make predictions or decisions without being explicitly programmed to do so. It enables computers to learn patterns from data and utilize that knowledge to perform tasks or make predictions.

### Explanation with Example

Imagine you want to build a program that can predict whether an email is spam or not spam (ham). Instead of manually writing rules to classify emails, you can use machine learning.

1. **Data Collection:** Gather a dataset of emails, each labeled as spam or not spam (ham).
   
2. **Feature Extraction:** Extract features from the emails, such as word frequency, presence of specific phrases, or email length.

3. **Training Phase:** Feed the labeled dataset into a machine learning algorithm. The algorithm learns from the data, identifying patterns that differentiate spam from non-spam emails.

4. **Model Building:** After training, the algorithm builds a model representing learned patterns. The model captures the relationship between email features and their labels (spam or not spam).

5. **Prediction:** When a new email arrives, input its features into the trained model. The model analyzes these features and predicts whether the email is spam or not based on learned patterns.

6. **Evaluation and Improvement:** Evaluate the model's performance using a separate test dataset. Refine the model by adjusting parameters, using different algorithms, or gathering more data.

## 2. AI vs ML vs DL
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

## 3. Types Of Machine Learning
   ![image](https://github.com/dev-madhurendra/100-Days-Of-Machine-Learning/assets/68775519/179fff85-c819-4118-b14c-197439dfc676)

   1. **Supervised Learning**:
      - Supervised learning involves training a model on a labeled dataset, where each example in the dataset is paired with an input and an output label.
      - The model learns to map inputs to outputs based on the examples provided during training.
      - The goal is for the model to make predictions or decisions when presented with new, unseen data.
      - Examples:
        - **Classification**: Predicting a categorical label. For example, classifying emails as spam or not spam based on their content.
        - **Regression**: Predicting a continuous value. For example, predicting the price of a house based on features like size, location, and number of bedrooms.
   
   2. **Unsupervised Learning**:
      - Unsupervised learning involves training a model on an unlabeled dataset, where the model learns to find patterns or structure in the data without explicit guidance.
      - The model discovers hidden patterns, groupings, or relationships within the data.
      - There are no predefined output labels, and the model must find its own way to represent the data.
      - Examples:
        - **Clustering**: Grouping similar data points together. For example, clustering customers based on their purchasing behavior to identify market segments.
        - **Dimensionality Reduction**: Reducing the number of features in a dataset while preserving its important characteristics. For example, reducing the dimensionality of high-dimensional data like images or text.
        - **Association**: [Beer and Diapers: The Impossible Correlation](https://tdwi.org/articles/2016/11/15/beer-and-diapers-impossible-correlation.aspx) Case Study
   
   3. **Reinforcement Learning**:
      - Reinforcement learning involves training an agent to interact with an environment and learn the best actions to take to maximize some notion of cumulative reward.
      - The agent learns through trial and error, receiving feedback in the form of rewards or penalties for its actions.
      - The goal is to learn a policy—a mapping from states of the environment to actions—that maximizes the cumulative reward over time.
      - Examples:
        - **Game Playing**: Teaching an AI agent to play games like chess or Go by rewarding good moves and penalizing bad ones.
        - **Robotics**: Training a robot to perform tasks like navigating a maze or manipulating objects in the environment by rewarding successful actions.

## 4. Key differences between types of machine learning

   | Aspect                     | Supervised Learning                  | Unsupervised Learning                | Reinforcement Learning                  |
   |----------------------------|--------------------------------------|--------------------------------------|----------------------------------------|
   | Training Data              | Labeled data                         | Unlabeled data                       | Agent-environment interaction with rewards/punishments |
   | Learning Process           | Learns from input-output pairs       | Learns patterns or structure in data| Learns from feedback (rewards/penalties) for actions|
   | Objective                  | Predict or classify new data         | Discover hidden patterns or groupings| Learn optimal actions in an environment|
   | Examples                   | Classification, Regression           | Clustering, Dimensionality Reduction| Game playing, Robotics                   |
   | Feedback Mechanism         | Supervised (provided labels)         | None (no explicit guidance)          | Reward/Punishment based on actions       |
   | Goal                       | Minimize prediction error            | Maximize data representation         | Maximize cumulative reward over time    |
   
   In summary:
   - **Supervised Learning** uses labeled data to predict or classify new data based on input-output pairs.
   - **Unsupervised Learning** learns from unlabeled data to discover hidden patterns or structure in the data.
   - **Reinforcement Learning** involves learning optimal actions through interaction with an environment, receiving feedback in the form of rewards or penalties.
      

