## Multi-Face-Recognition
In this project, I use MTCNN for face detection and VGG16 for face recognition.

## Description
### Face Recognition (MTCNN)
In the realm of computer vision,Face Recognition serves the pivotal role of distinguishing and validating individuals through digital imagery or video snippets, extensively utilized in security infrastructure, biometric verifications, and social media interfaces.

### Face Recognition (VGG16)
VGG16, crafted by the Visual Geometry Group (VGG) at the University of Oxford, stands as a hallmark in the realm of convolutional neural network architectures introduced by Qassim, H., et al. in their paper **Compressed residual-VGG16 CNN model for big data places image recognition**. Renowned for its simplicity and efficacy, this model comprises 16 layers, characterized by a series of convolutional layers alternated with max-pooling layers. Its design, though deep, maintains a straightforward structure, making it a favored choice for both academic research and practical applications.  

### Classifying Images with VGG16 Using Various Machine Learning Algorithms

Logistic Regression: By fitting a logistic function to the extracted features, the model assigns probabilities to each class, facilitating robust and interpretable image classification tasks with VGG16 embeddings.

Linear SVM: Leveraging the high-dimensional feature space extracted by VGG16, the linear SVM effectively delineates decision boundaries, enabling accurate image classification across diverse datasets.

Neural Network: By combining the hierarchical representations learned by VGG16 with additional layers, the neural network can capture complex patterns and relationships, achieving state-of-the-art performance in image classification benchmarks.

Gradient Boosting: By sequentially fitting decision trees to the residual errors of previous models, gradient boosting effectively combines the predictive power of VGG16 embeddings with the flexibility of tree-based learners, yielding robust and accurate image classification results.

Nearest Neighbor: By measuring distances or similarities between feature vectors, nearest neighbor classifiers offer a simple yet powerful approach to image classification, capable of capturing complex data distributions and achieving competitive performance with minimal assumptions.


## Table of Contents
- [Installation](#installation)
- [Author](#author)
- [License](#license)

## Installation
1. Clone the repository:
```
git clone https://github.com/WaiHninEaindrarMg/Multi-Face-Recognition.git
```

2. Install MTCNN :

```
pip install pytorch-mtcnn
```

## Instruction
1. Run this file if you have images https://github.com/WaiHninEaindrarMg/Multi-Face-Recognition/face_detection_image.py or
   videos https://github.com/WaiHninEaindrarMg/Multi-Face-Recognition/face_detection_video.py
```
python face_detection_image.py or python face_detection_video.py 
```

2. Run this file Train.ipynb
```
Run Train.ipynb
```
In this Train.ipynb , There is vgg16 model for training.
This is performace plot for train and validation accuracy classify with 5 machine learning algorithms.
![Accuracy](confusion_matrix/LR1.png)


3. Run this file face_recognition.py
```
Run face_recognition.py
```
After run this face_recognition.py, video output will be showed.
This is result video for vgg16 model with Linear SVM classification (face recognition result)
![Result]()

##
## Author
👤 : Wai Hnin Eaindrar Mg  
📧 : [waihnineaindrarmg@gmail.com](mailto:waihnineaindrarmg@gmail.com)


## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE.md file for details.
