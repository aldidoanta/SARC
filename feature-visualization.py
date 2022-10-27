import matplotlib.patches as mpatches

# Plot the training samples
idx = (distID_train == 0)  # select the samples from the first distribution centered in (0,0)
pl.scatter(X_train[idx, 0], X_train[idx, 1],
           c=y_train[idx], marker='^',
           cmap=pl.cm.Paired, s=100, edgecolors='k')

# ~idx contains the indices of the samples from distribution 2
pl.scatter(X_train[~idx, 0], X_train[~idx, 1],
           c=y_train[~idx], marker='o',
           cmap=pl.cm.Paired, s=100, edgecolors='k')

# Plot the ground truth ranking line
x_space = np.linspace(X_train[:,0].min(), X_train[:,0].max())
pl.plot(x_space * w[0], x_space * w[1] + X_train[:,1].mean(), color='gray')

cm = pl.cm.get_cmap('Paired')
pl.legend(loc='lower right',
          handles=[mpatches.Patch(facecolor=cm(0.), edgecolor='k', label='Rank 0'),
                   mpatches.Patch(facecolor=cm(0.5), edgecolor='k', label='Rank 1'),
                   mpatches.Patch(facecolor=cm(1.), edgecolor='k', label='Rank 2'),])

#pl.axis('equal')
pl.title('Training data')
pl.show()