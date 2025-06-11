# ML Playground

## Projects

- [`01-lin-reg-api`](./01-lin-reg-api): Linear regression API with FastAPI and PyTorch, featuring manual SGD and prediction endpoints.
- [`02-log-reg-api`](./02-log-reg-api): Logistic regression API using FastAPI and PyTorch for binary classification.
- [`03-softmax-reg-api`](./03-softmax-reg-api): Multiclass softmax regression API built using PyTorch, with manual cross-entropy loss, gradient updates, and FastAPI endpoint for training and prediction. Includes a **quick & dirty manual MNIST test** (`mnist_test.py`) to benchmark the softmax regression model. Currently achieves **~70% accuracy after 30 epochs on 10k samples**. Needs optimization.
- [`04-gda-banknote`](./04-gda-banknote): Gaussian Discriminant Analysis (GDA) trained on the real-world Banknote Authentication dataset. Includes per-class covariance estimation, feature normalization, and evaluation via confusion matrix and accuracy. 