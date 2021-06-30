import os
import seaborn as sns
import matplotlib.pyplot as plt


def graph_train_data(data, path, step = 1):
    # Reduce the density of the data
    if step > 1:
        data = data.iloc[::step, :]
    # Make sure that the folders the file is meant to be in exist
    if not os.path.exists(path[:path.rindex("/")]):
        os.makedirs(path[:path.rindex("/")])
    # Plot the data
    plt.clf()
    sns.set_theme(style="darkgrid")
    ax = sns.lineplot(data = data, x = "epoch", y = "acc")
    sns.set_theme(style="dark")
    ax2 = ax.twinx()
    sns.lineplot(data = data, x = "epoch", y = "loss", ax = ax2, color = "orange")
    plt.savefig(path)
