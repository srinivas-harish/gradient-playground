import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Normalize features
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
 
mu = np.array([X[y == c].mean(axis=0) for c in range(3)])
sigma = np.array([np.cov(X[y == c].T) for c in range(3)])
 
def generate_flower_features(class_id, n=5):
    return np.random.multivariate_normal(mu[class_id], sigma[class_id], n)
 
def draw_flower(features, class_id, idx):
    sepal_len, sepal_wid, petal_len, petal_wid = features
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
 
    for angle in [0, np.pi]:
        ax.plot(
            [0, sepal_len * 0.2 * np.cos(angle)],
            [0, sepal_wid * 0.2 * np.sin(angle)],
            color='green', lw=3
        )
 
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        ax.plot(
            [0, petal_len * 0.3 * np.cos(angle)],
            [0, petal_wid * 0.3 * np.sin(angle)],
            color='purple', lw=5, alpha=0.8
        )
 
    fname = f"flower_class{class_id}_{idx}.png"
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved flower image: {fname}")
 
for class_id in range(3):
    samples = generate_flower_features(class_id, n=5)
    for i, sample in enumerate(samples):
        draw_flower(sample, class_id, i)
