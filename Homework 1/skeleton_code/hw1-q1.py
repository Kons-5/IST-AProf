#!/usr/bin/env python

# Deep Learning Homework 1

import os

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils

# ========================
# Aux FUNCTIONS
# ========================

def log_softmax(x):
    """
    Compute the softmax probabilities for a vector of scores
    While avoiding taking the log of zero
    """
    x_max = np.max(x)      
    return x - x_max - np.log(np.sum(np.exp(x - x_max)))

def softmax(scores):
    """
    Compute the softmax probabilities for a vector of scores
    """
    exp_scores = np.exp(scores - np.max(scores)) # Prevent overflow (Check)
    return exp_scores / np.sum(exp_scores)

def cross_entropy(y_hat, y_one_hot):
    """
    Compute the cross-entropy loss using log-softmax for stability.
    """
    log_probs = log_softmax(y_hat)
    return -np.dot(y_one_hot, log_probs)

def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

# ========================
# CLASSES 
# ========================

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """
        X (n_examples x n_features)
        """
        scores = np.dot(self.W, X.T)              # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        
        # Compute the scores for all classes
        scores = np.dot(self.W, x_i)  
        
        # Predict the class with the highest score
        y_hat = np.argmax(scores)
        
        # Update weights if prediction is incorrect
        if y_hat != y_i:
            self.W[y_i] += x_i          # Increase weight for the true class
            self.W[y_hat] -= x_i        # Decrease weight for the incorrect predicted class


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Compute the scores for all classes
        scores = np.dot(self.W, x_i)  
        
        # Compute probabilities using the softmax function 
        probabilities = softmax(scores)  
        
        # Create a one-hot vector for the true label
        one_hot = np.zeros(self.W.shape[0])  
        one_hot[y_i] = 1                                    # Set the true class index to 1
        
        # Compute the gradient for each class
        gradient = np.outer(probabilities - one_hot, x_i)  

        # Add L2 regularization to the gradient
        gradient += l2_penalty * self.W                     # If l2_penalty = 0.0 defaults to 
                                                            #non-regularized version of the classifier.
        
        # Update the weights with gradient descent
        self.W -= learning_rate * gradient


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        """
        Initialize weights and biases for a single hidden layer MLP.

        Sizes:
        W1:     (hidden_size, n_features)
        W2:     (n_classes, hidden_size)
        b1:     (hidden_size, )
        b2:     (n_classes, )
        """
        
        self.n_classes = n_classes
        
        # Input to hidden layer weights and biases
        self.W1 = np.random.normal(loc=0.1, scale=0.1, size=(hidden_size, n_features))  
        self.b1 = np.zeros(hidden_size) 

        # Hidden to output weights and biases
        self.W2 = np.random.normal(loc=0.1, scale=0.1, size=(n_classes, hidden_size))  
        self.b2 = np.zeros(n_classes)  


    def forward(self, X, save_hidden=False):
        """
        Perform the forward pass of the MLP.
        
        Sizes:
        X:     (n_examples, n_features)
        """
        # Hidden layer pre-activation
        z1 = np.dot(X, self.W1.T) + self.b1   # (n_examples,hidden_size)
        
        # Hidden layer activation (ReLU)
        h1 = np.maximum(0, z1)                # (n_examples,hidden_size)
        
        # Output layer pre-activation
        z2 = np.dot(h1, self.W2.T) + self.b2  # (n_examples, n_classes)
        
        # Return hidden activations only for backpropagation
        return z2, h1 if save_hidden else None


    def backward(self, x, y, probs, h1, learning_rate=0.001):
        """
        Perform backward propagation and update weights and biases for SGD.
        """
        
        # Gradients for output layer
        grad_z2 = probs - y                     # (n_classes,)
        grad_W2 = np.outer(grad_z2, h1)         # (n_classes, hidden_size)
        grad_b2 = grad_z2                       # (n_classes,)
    
        # Backpropagate to hidden layer
        grad_h1 = np.dot(self.W2.T, grad_z2)    # (hidden_size,)
        grad_z1 = grad_h1 * (h1 > 0)            # Apply ReLU derivative, (hidden_size,)
    
        # Gradients for input layer
        grad_W1 = np.outer(grad_z1, x)          # (hidden_size, n_features)
        grad_b1 = grad_z1                       # (hidden_size,)
    
        # Update weights and biases
        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1
        self.W2 -= learning_rate * grad_W2
        self.b2 -= learning_rate * grad_b2


    def predict(self, X):
        """
        Compute the forward pass of the network. At prediction time, there is
        no need to save the values of hidden nodes.
        
        Sizes:
        X:     (n_examples, n_features)    
        """
        
        # Perform forward pass 
        logits, _ = self.forward(X)        
    
        # Return predicted class indices
        return np.argmax(logits, axis=1)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Train the model for one epoch using SGD and
        return the loss of the epoch.
        """
        
        n_examples = y.shape[0]
        loss = 0

        # Process batch sizes = 1
        for x, label in zip(X, y): 
            
            # create one-hot encoding for the label
            y_one_hot = np.zeros(self.n_classes)     
            y_one_hot[label] = 1 
    
            # Compute Forward pass
            output, h = self.forward(x, True)

            # Compute softmax probabilities for backpropagation
            probs = softmax(output)
    
            # Compute cross-entropy loss using log-softmax
            loss += cross_entropy(output, y_one_hot)
    
            # Compute Backward pass to update weights and biases
            self.backward(x, y_one_hot, probs, h, learning_rate)

        # Average the loss across all examples
        loss /= n_examples
        return loss

# ========================
# MAIN 
# ========================

def main():

    # ===========================
    # ARGUMENT PARSING
    # ===========================
    
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument(
        'model',
        choices=['perceptron', 'logistic_regression', 'mlp'],
        help="Which model should the script run? Options: perceptron, logistic_regression, mlp."
    )
    parser.add_argument(
        '-epochs', default=20, type=int,
        help="Number of epochs to train for. Default is 20."
    )
    parser.add_argument(
        '-hidden_size', type=int, default=100,
        help="Number of units in hidden layers (only for MLP). Default is 100."
    )
    parser.add_argument(
        '-learning_rate', type=float, default=0.001,
        help="Learning rate for parameter updates (only for logistic regression and MLP). Default is 0.001."
    )
    parser.add_argument(
        '-l2_penalty', type=float, default=0.0,
        help="L2 regularization strength. Default is 0 (no regularization)."
    )
    parser.add_argument(
        '-data_path', type=str, default='intel_landscapes.v2.npz',
        help="Path to the dataset file. Default is 'intel_landscapes.v2.npz'."
    )
    opt = parser.parse_args()

    # ===========================
    # DATA LOADING 
    # ===========================

    utils.configure_seed(seed=42)  
    
    add_bias = opt.model != "mlp"  
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  
    n_feats = train_X.shape[1]  

    utils.configure_seed(seed=42)

    # ===========================
    # INIT MODEL
    # ===========================

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)

    # ===========================
    # TRAINING SETUP
    # ===========================
    
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))

        # Shuffle training data
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
            
        # Evaluate performance on training and validation sets
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))

    # ===========================
    # TRAINING SUMMARY
    # ===========================
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # ===========================
    # PLOTTING
    # ===========================

    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    
    accuracy_plot_file = os.path.join(results_dir, f"Q1-{opt.model}-accs.pdf")
    plot(
        epochs,
        train_accs,
        valid_accs,
        filename=accuracy_plot_file
    )
    
    if opt.model == 'mlp':
        loss_plot_file = os.path.join(results_dir, f"Q1-{opt.model}-loss.pdf")
        plot_loss(
            epochs,
            train_loss,
            filename=loss_plot_file
        )
    elif opt.model == 'logistic_regression':
        norm_plot_file = os.path.join(results_dir, f"Q1-{opt.model}-w_norms.pdf")
        plot_w_norm(
            epochs,
            weight_norms,
            filename=norm_plot_file
        )
        
    results_file = os.path.join(results_dir, f"Q1-{opt.model}-results.txt")
    

    with open(results_file, "w") as f:
        final_test_acc = model.evaluate(test_X, test_y)
        f.write(f"Final test acc: {final_test_acc:.4f}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")
    

if __name__ == '__main__':
    main()
