# DEEP-LEARNING-PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: GURRALA LAXMI PARIMAL

INTERN ID: CT12RER

DURATION: 8 WEEKS

MENTOR: NEELA SANTHU

##Description:This Python script implements a deep learning workflow to classify images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, including airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset is split into 50,000 training images and 10,000 test images. The objective of this code is to train a model that can accurately identify the class of a given image from the dataset. The task is a classic example of supervised image classification, where the model learns to map input images to their respective category labels.
The script begins by importing essential libraries: TensorFlow and its Keras API for model building, NumPy for numerical operations, and Matplotlib for visualizations. The dataset is loaded using keras.datasets.cifar10.load_data(), which automatically downloads and splits the dataset into training and testing sets. Each image’s pixel values are normalized by dividing by 255.0 to scale them into the [0,1] range, which helps the model train more efficiently and avoid issues caused by large input values. A list of class names is defined to translate numeric labels into human-readable categories, which is used later for visualization and interpretation of model predictions.
The CNN architecture is defined using Keras’s Sequential API. It includes three convolutional layers (Conv2D) with increasing filter sizes (32, 64, and 128) to capture complex patterns in the images, each followed by a max pooling layer (MaxPooling2D) that reduces the spatial dimensions and helps with overfitting. After the convolutional layers, a Flatten layer transforms the 3D feature maps into a 1D vector, which is then fed into a dense hidden layer with 128 units and ReLU activation. Finally, the output layer has 10 neurons, corresponding to the 10 CIFAR-10 classes, and uses the softmax activation function to output a probability distribution across all classes.
The model is compiled with the Adam optimizer, which is efficient for training deep neural networks, and uses sparse categorical crossentropy as the loss function, appropriate for multi-class classification with integer labels. Accuracy is used as the evaluation metric. The model is trained for 10 epochs on the training data, and its performance is validated on the test data during each epoch to monitor generalization.
Once training is complete, the script evaluates the model on the test set using model.evaluate() and prints the resulting accuracy. It also visualizes the training and validation accuracy over the epochs using Matplotlib, which provides insight into whether the model is learning effectively or overfitting. Finally, it generates predictions for the first 10 test images and displays them along with their predicted and true labels. The label color indicates prediction correctness—green for correct predictions and red for incorrect.
This script is intended to run in any Python environment that supports TensorFlow, such as local machines, Google Colab, or cloud platforms like AWS or Azure. It demonstrates an end-to-end machine learning workflow, including data preparation, model design, training, evaluation, and result visualization using powerful tools from the Python ecosystem.
