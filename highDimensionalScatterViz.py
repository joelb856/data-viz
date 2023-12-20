import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class highDimensionalScatterViz():

    def __init__(self,df) -> None:
        self.df = df

    @staticmethod
    def distance_from_median(df):
        median = df.median(axis=0)
        nrows = df.shape[0]

        median_matrix = np.transpose(np.vstack([median]*nrows))
        coordinates_matrix = np.transpose(df)
        distances = np.linalg.norm(median_matrix - coordinates_matrix, axis=0)

        return distances

    def projected_scatter(self):

        print('Making cool plots...')
        df = self.df

        distances = self.distance_from_median(df)
        df.iloc[distances.argsort()]

        n_dimensions = len(df.columns) - 1
        fig, ax = plt.subplots(n_dimensions, n_dimensions, figsize=(10,10))

        for i in range(n_dimensions):

            y_name = df.columns[i + 1]
            for j in range(n_dimensions):

                if (j <= i):

                    x_name = df.columns[j]
                    x = df[x_name]
                    y = df[y_name]
                    # xy = np.vstack([x,y])
                    # z = gaussian_kde(xy)(xy)
                    # idx = z.argsort()
                    # x, y, z = x[idx], y[idx], z[idx]

                    ax[i,j].scatter(x, y, s=3, c=distances, cmap='viridis_r')

                    ax[i,j].set_xticks([])
                    ax[i,j].set_yticks([])

                    if (i == n_dimensions - 1):
                        ax[i,j].set_xlabel(x_name)
                    if (j == 0):
                        ax[i,j].set_ylabel(y_name)
            
                else:
                    ax[i,j].axis('off')
        plt.subplots_adjust(hspace=.0)
        plt.subplots_adjust(wspace=.0)
        plt.show()

    @staticmethod
    def blurred_scatter(x,y,xerr,yerr,npoints=10000):
        
        plt.figure()
    
        for i in range(len(x)):
            xdraws = np.random.normal(x[i],xerr[i],npoints)
            ydraws = np.random.normal(y[i],yerr[i],npoints)
            xy = np.vstack([xdraws,ydraws])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            xdraws, ydraws, z = xdraws[idx], ydraws[idx], z[idx]
            plt.scatter(xdraws, ydraws, s=1, c=z)

        plt.show()
