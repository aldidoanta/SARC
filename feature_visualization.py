import matplotlib as mpl
import pylab as pl

def plot_sentiment_features(x_train, y_train):
    # Plot the training samples
    compound_previous = [item[0] for item in x_train]
    compound_current = [item[1] for item in x_train]

    pl.scatter(compound_previous, compound_current,
               c=y_train, marker='o',
               cmap=mpl.colors.ListedColormap(['blue', 'orange']),
               norm=mpl.colors.Normalize(vmin=-1, vmax=1),
               edgecolors='k')

    cm = mpl.colors.ListedColormap(['blue', 'orange'])
    pl.legend(loc='lower right',
              handles=[mpl.patches.Patch(facecolor=cm(0.), edgecolor='k', label='Non-sarcastic'),
                       mpl.patches.Patch(facecolor=cm(1.), edgecolor='k', label='Sarcastic')])

    pl.title('Training Data')
    pl.show()