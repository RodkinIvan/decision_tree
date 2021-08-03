import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification
import seaborn as sns


classes_num = 2
dataset = make_circles(noise=0.09, factor=0.5, random_state=42)

palette = sns.color_palette(n_colors=classes_num)
cmap = ListedColormap(palette)

x, y = dataset
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap, alpha=.8)


def plot_surface(clf, X, y):
    plot_step = 0.01
    palette = sns.color_palette(n_colors=len(np.unique(y)))
    cmap = ListedColormap(palette)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()].tolist())
    Z = np.asarray(Z).reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=.7,
                edgecolors=np.array(palette)[y], linewidths=2)


from decision_tree import decision_tree_classifier


clf = decision_tree_classifier(classes_num)
clf.fit(x.tolist(), y.tolist())
plot_surface(clf, x, y)
plt.show()



