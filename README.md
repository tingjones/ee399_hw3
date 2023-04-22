# ee399_hw3

This assignment involved evaluating the different classification methods on the MNIST dataset. This dataset contains 70,000 images of 28x28 images of a single handwritten digit, with the images labeled for what number they represent. Each of the images therefore have 784 features (28x28). Due to hardware limitations, a random sample of 4000 was extracted from the dataset and operations were performed on this subset instead.

A sample of the MNIST dataset is given in Fig. 1 with their corresponding labels.

Firstly, a sample subset was taken from the 70,000 image size dataset to speed computations. Each image and its features were assigned to one vector, X, and then transposed so that each column represneted a single image. Each images label, or target, was assigned to y, which was a 1D matrix containing the labels as type string.

This task involved performing SVD on the dataset, which will factorize the matrix into matrices U, S, and V^T. U is the unitary matrix of left side eigenvectors. S represents np.diag(s), where s^2 are the eigenvalues. V^T are the right side eigenvectors, which are transposed. 

The objective of this task is to deterkine the rank of the basis matrix and then plot the Singular Value Spectrum. This involved plotting the eigenvalue magnitude against the Principle Component Number. This was plotted on the logarithmic scale to better visualize when the magnitude approaches 0. As seen here, the rank was found to be 642, which has the eignvalue magnitude of around 3x10^-26, very close to 0. It is also after a severe drop in the eigenvalue magnitude.

The next task involved selecting three digits and plotting the projection of the dataset on the corresponding V mode vectors, then, plotting these projections on a 3D scatterplot. The code for finding the projection is given below.

The objective of this task is to pick two digits and attempt to split the training dataset using lines. This was done using Support Vector Machines (SVM), which find lines that cut between the datapoints and attempt high accuracy in separating points of different labels. In other words, it optimally divides the data into classes.

Using the scikit package, the Support Vector Classifier is built through the SVC() module, and with the additional argument of "linear," the SVM will attempt to fit lines (or planes in 3D) between the data.

The results can be plotted on the scatterplot with the original training datapoints and the support vectors can be highlighted to convey where the lines to split the data would be.

To evaluate the accuracy of the model, results of the fit's predictions on the datapoint's class were compared to the ground truth labels for each of the datapoints.

The objective for this problem was to repeat the process in problem 5, but to now select 3 digits and attempt to classify between them. This was again done with the SVC module and fitting with lines (planes), but instead attempting to split between three classes within the data.