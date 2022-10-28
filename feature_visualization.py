import matplotlib as mpl
import pylab as pl

def plot_sentiment_features(docs, labels, plot_title, marker):
    # Plot the training samples
    compound_previous = [item[0] for item in docs]
    compound_current = [item[1] for item in docs]

    figure = pl.figure(dpi=150)
    pl.scatter(compound_previous, compound_current,
               c=labels, marker=marker,
               cmap=mpl.colors.ListedColormap(['black', 'orange']),
               norm=mpl.colors.Normalize(vmin=-1, vmax=1),
               edgecolors='k')

    cm = mpl.colors.ListedColormap(['black', 'orange'])
    pl.legend(loc='lower right',
              handles=[mpl.patches.Patch(facecolor=cm(0.), edgecolor='k', label='Non-sarcastic'),
                       mpl.patches.Patch(facecolor=cm(1.), edgecolor='k', label='Sarcastic')])

    pl.title(plot_title)
    pl.xlabel("Previous sentiment's compound score")
    pl.ylabel("Current sentiment's compound score")

    figure.savefig(plot_title + '.png')
