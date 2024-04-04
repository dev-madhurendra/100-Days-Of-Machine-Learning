## Table of Contents
1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [AI vs ML vs DL](#2-ai-vs-ml-vs-dl)
3. [Types of Machine Learning](#3-types-of-machine-learning)_
4. [Key differences between types of machine learning](#4-key-differences-between-types-of-machine-learning)
5. [Batch Machine Learning vs Online Machine Learning](#5-batch-machine-learning-vs-online-machine-learning)

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
        - **Dimensionality Reduction**: Reducing the number of features in a dataset while preserving its important characteristics. For example, reducing the dimensionality of high-dimensional data like images or text [Blog](https://colah.github.io/posts/2014-10-Visualizing-MNIST/).
        - **Association**: [Beer and Diapers: The Impossible Correlation](https://tdwi.org/articles/2016/11/15/beer-and-diapers-impossible-correlation.aspx) Case Study
   
   3. **Reinforcement Learning**:
      - Reinforcement learning involves training an agent to interact with an environment and learn the best actions to take to maximize some notion of cumulative reward.
      - The agent learns through trial and error, receiving feedback in the form of rewards or penalties for its actions.
      - The goal is to learn a policy—a mapping from states of the environment to actions—that maximizes the cumulative reward over time.
      - Examples:
        - **Game Playing**: Teaching an AI agent to play games like chess or Go [Full Documentry on Go by Google DeepMind](https://www.youtube.com/watch?v=WXuK6gekU1Y) by rewarding good moves and penalizing bad ones.
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
      
## 5. Batch Machine Learning vs Online Machine Learning
   ![image](https://github.com/dev-madhurendra/100-Days-Of-Machine-Learning/assets/68775519/63ba54b2-8496-4bd4-a5b4-92b312a121be)

   ### Batch Learning : 
   Batch learning, also known as batch processing or offline learning, is a machine learning approach where the model is trained using the entire dataset at once. In batch learning, the    algorithm processes the entire dataset, computes the gradients, and updates the model parameters all at once. Here's a breakdown of batch learning:

   1. **Data Processing**:
      - The entire dataset is loaded into memory and processed as a single batch.
      - This means that all the data points, features, and labels are available to the algorithm simultaneously.
   
   2. **Model Training**:
      - The model is trained on the entire dataset for multiple iterations or epochs.
      - During each epoch, the algorithm goes through the entire dataset and updates the model parameters based on the computed gradients.
   
   3. **Parameter Updates**:
      - After each epoch, the model's parameters are updated based on the computed gradients of the entire dataset.
      - This involves adjusting the weights and biases of the model to minimize the loss function, which measures the difference between the predicted outputs and the actual labels.
   
   4. **Evaluation**:
      - Once the model has been trained, it is evaluated on a separate validation dataset to assess its performance.
      - The model's accuracy, precision, recall, or other relevant metrics are calculated to determine its effectiveness in making predictions on unseen data.
   
   **Advantages of Batch Learning**:
   - **Simplicity**: Batch learning is straightforward to implement and understand, making it suitable for beginners and for scenarios where simplicity is preferred.
   - **Stability**: The model parameters are updated less frequently compared to online learning, resulting in stable learning and convergence.
   
   **Disadvantages of Batch Learning**:
   - **Memory Intensive**: Batch learning requires loading the entire dataset into memory, which can be challenging for large datasets that may not fit into memory.
   - **Lack of Adaptability**: Batch learning is unable to adapt to changes in the data distribution without retraining the entire model, making it less suitable for dynamic or evolving datasets.
   
   Batch learning is commonly used in scenarios where the entire dataset is available upfront, and real-time predictions are not required. It is well-suited for tasks such as offline data analysis, batch processing of large datasets, and model training in research and development environments.

### Online Machine Learning
   Online machine learning, also known as incremental learning or online learning, is a machine learning approach where the model is continuously updated with new data points as they become available. Unlike batch learning, where the model is trained using the entire dataset at once, online learning updates the model incrementally, typically on small batches or individual data points. Here's an overview of online machine learning:

1. **Incremental Updates**:
   - In online learning, the model is updated continuously as new data points arrive.
   - Instead of processing the entire dataset at once, the model processes incoming data points sequentially or in small batches.

2. **Training Approach**:
   - The model learns from new data points by updating its parameters incrementally.
   - Each new data point or batch of data points is used to adjust the model's parameters, such as weights and biases, to improve its performance.

3. **Adaptability**:
   - Online learning algorithms are more adaptable to changes in the data distribution over time.
   - The model can quickly adjust to new patterns or trends in the data without requiring retraining on the entire dataset.

4. **Scalability**:
   - Online learning is suitable for handling large datasets that may not fit into memory or may be too computationally expensive to process at once.
   - By processing data incrementally, online learning algorithms can efficiently scale to handle streaming data or continuous data streams.

5. **Real-time Predictions**:
   - Online learning allows for real-time predictions, as the model can update and make predictions on new data points as they arrive.
   - This makes online learning suitable for applications that require immediate feedback or decision-making, such as fraud detection, recommendation systems, and online advertising.

**Advantages of Online Learning**:
- **Adaptability**: Online learning algorithms can adapt to changes in the data distribution over time, making them suitable for dynamic or evolving datasets.
- **Scalability**: Online learning is efficient for processing large datasets or streaming data, as it can handle data incrementally without requiring the entire dataset to be loaded into memory.

**Disadvantages of Online Learning**:
- **Complexity**: Online learning algorithms can be more complex to implement compared to batch learning, as they require handling data streams and managing model updates.
- **Sensitivity to Noise**: Online learning algorithms may be sensitive to noisy or irrelevant data points, requiring careful preprocessing and handling of data.

Online learning is commonly used in scenarios where data arrives continuously or in real-time, and where immediate feedback or predictions are required. It is well-suited for applications such as online advertising, recommendation systems, sensor data analysis, and anomaly detection.


| Aspect                    | Batch Machine Learning         | Online Machine Learning       |
|---------------------------|--------------------------------|-------------------------------|
| Training Approach         | Trains the model using the entire dataset at once. | Updates the model incrementally with new data points as they become available. |
| Data Processing           | Processes the entire dataset as a single batch. | Processes data sequentially or in small batches, typically on streaming or continuous data. |
| Adaptability              | Less adaptable to changes in the data distribution without retraining. | More adaptable to changes in the data distribution over time, as the model continuously updates with new data. |
| Memory Usage              | Requires loading the entire dataset into memory, memory-intensive. | Can process data incrementally, suitable for handling large datasets or streaming data. |
| Model Stability           | Stable learning, as model parameters are updated less frequently. | Prone to overfitting and sensitivity to noisy or irrelevant data points. |
| Real-time Predictions     | Typically used for offline analysis and batch processing. | Suitable for real-time predictions and applications requiring immediate feedback. |
