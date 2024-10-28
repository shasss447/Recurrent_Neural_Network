# Simple RNN from Scratch using Numpy

This project implements a simple Recurrent Neural Network (RNN) from scratch, using only Numpy. It demonstrates the core concepts behind RNNs, including one-hot encoding, forward propagation, and backpropagation. This implementation was created as an exercise in understanding the mechanics of RNNs at the fundamental level, without the use of high-level machine learning libraries.

## Project Description

In this project, a simple RNN model is built to classify sequences of words. The model uses only Numpy to perform all operations, from vector encoding to the training process, and includes:
- **Vector Encoding**: Words are first converted into vectors by collecting all unique words in the dataset. Each word is then represented using a one-hot encoding scheme.
- **RNN Architecture**: The model consists of a single neuron with 3 weights and 2 biases that form the RNN cell.
- **Backpropagation**: The model includes a custom implementation of backpropagation, enabling the network to learn from errors and improve its predictions.

## Features

- **One-hot Encoding**: Convert words into a unique vector representation.
- **Manual Forward Propagation**: Computes the hidden states and output from scratch.
- **Backpropagation from Scratch**: Gradient calculation and updates are manually implemented to adjust weights and biases.

## Dataset

The model uses sample training and testing datasets for demonstration purposes. The data is in the form of key-value pairs, where:
- The keys represent input sentences (sequences of words).
- The values represent the target classes (sentiment labels or categories).

## Project Structure

* **`main.py`**: Main script that runs the RNN training and testing process.
* **`data.py`**: Contains the training and testing data in key-value format.
* **`rnn.py`**: Core RNN implementation, including forward and backward propagation, as well as gradient clipping.

## Code Structure

### Forward Propagation

In each forward pass, the RNN performs the following steps:
1. Initialize the hidden state (`h`).
2. Loop through the input sequence, updating the hidden state at each step.
3. Compute the output of the RNN based on the final hidden state.

### Backpropagation

Backpropagation is implemented from scratch, involving:
1. Calculation of the gradients with respect to weights and biases.
2. Propagation of gradients through the sequence in reverse.
3. Gradient Clipping: Limits the gradients to a maximum value to prevent exploding gradients.
4. Weight and bias updates at each training step.


### Performance Metrics

The model computes the following metrics at each epoch:
- **Loss**: Cross-entropy loss to measure the model's performance.
- **Accuracy**: Percentage of correctly classified samples.

### Results

The model was trained over 1000 epochs, achieving high accuracy on both the training and testing sets. Below is the graph showing loss and accuracy over epochs:

![Training and Testing Accuracy and Loss](/train_test_accuracy_loss.png)

* **Final Training Accuracy**: 100%
* **Final Testing Accuracy**: 100%



