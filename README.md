# The Enhancement and Automated Counting of Bats

# PyTorch Image Classification Project

This project implements a PyTorch-based image classification pipeline using the ResNet-18 architecture for multiple image classification models. Each model is trained on different subsets of the dataset (1-4, 5-8, 9-12, 1-12), and an additional top-level model is trained to distinguish between these subsets.

## Overview: The Enhancement and Automated Counting of Bats

This project focuses on the ecological importance of bats by enhancing the accuracy of bat counting in small, low-resolution images using convolutional neural networks (CNNs). Bats play a crucial role in ecosystems, contributing to pollination, seed dispersal, and insect population control. Accurate population assessments of bats are essential for ecological research and conservation strategies.

Our study aims to overcome the challenges posed by low-quality images, such as blurriness, background clutter, and overlapping bats, which traditionally lead to inaccurate counts and misinterpretations. By leveraging advanced machine learning techniques, we have developed a robust and accurate model to detect and count bats in these challenging images.

### Features

- **Convolutional Neural Networks (CNNs):** The project utilizes a two-tiered CNN architecture to detect and count bats. The model is capable of handling images with varying numbers of bats, ranging from 1 to 12.
  
- **Synthetic Data Augmentation:** To address class imbalances and improve the model's accuracy, synthetic images were generated to simulate a variety of realistic scenarios.

- **Ensemble Methodology:** The ensemble model enhances the overall accuracy by using sub-models to focus on different subsets of the bat count problem.

- **Real-world Testing:** The model was validated through real-world applications, where automated predictions were compared against manual counts by professionals.

### Results

The model demonstrates a significant improvement in bat detection and counting accuracy, achieving an overall accuracy of 93% with the ensemble model. It was particularly effective in counting smaller groups of bats but faced challenges with larger and more complex groups. The synthetic dataset and CNN architecture proved to be robust tools for this task.

### Future Work

Future research could focus on enhancing the dataset with more diverse scenarios and different bat species. Additionally, incorporating Region-based CNN (R-CNN) could improve the model’s spatial detection capabilities, further refining bat counting accuracy.

### Contributors

- **Zhi Zheng** - Data Science, Florida Polytechnic University
- **Benjamin Bowman** - Computer Science, Florida Polytechnic University
- **Nesreen Dalhy** - Computer Science, Florida Polytechnic University
- **Brendan Geary** - Computer Science, Florida Polytechnic University
- **Bayazit Karaman** - Computer Science, Florida Polytechnic University
- **Ian Bentley** - Physics, Florida Polytechnic University

## Project Structure

- `Final_Project_Training_1_1.ipynb`: The main notebook where the model is trained, validated, and tested. It includes the data preprocessing, model training, and performance evaluation.
- `random_placement_2.ipynb`: A supporting notebook that deals with the synthetic data generation and random placement of bats in images.
- `The Enhancement and Automated Counting of Bats.pdf`: The project report that details the methodology, results, and discussion of the study.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- tqdm (for progress bars)

### Installation

First, install the required dependencies:

```bash
pip install torch torchvision tqdm
```

## Dataset Structure

The dataset should be organized into folders where each subfolder represents a class. The project expects the dataset to be divided into different clusters (e.g., `1_4`, `5_8`, etc.) as well as a top-level folder for the overall classification.

Example folder structure:

```
/content/Final Testing Dataset
├── 1_12
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── ...
├── Top_level
│   ├── 1_4
│   ├── 5_8
│   ├── 9_12
```

The dataset for each model (`1_4`, `5_8`, `9_12`, and `1_12`) should be organized similarly.

## How the Code Works

1. **Config Class**: Contains configuration options for device selection (GPU/CPU), data paths, model save paths, number of classes, training parameters, and learning rates for each model.

2. **Data Loading**: The `torchvision.datasets.ImageFolder` is used to load the dataset, and data is transformed using `torchvision.transforms` (resize, normalization, etc.).

3. **Model Architecture**: A modified ResNet-18 model is used for classification. The fully connected layer (`fc`) is replaced with a custom layer that matches the number of output classes for each model. Dropout is added to reduce overfitting.

4. **Training and Validation**: Each model is trained using the Adam optimizer with a cross-entropy loss function. A learning rate scheduler reduces the learning rate after a set number of epochs, and early stopping is implemented to halt training if validation loss does not improve for a specified number of epochs.

5. **Saving and Loading**: After training, each model is saved to disk. If a model exists, it can be loaded instead of retraining.

## How to Run the Code

### 1. Prepare the Dataset

Ensure that the dataset is properly organized into the specified folder structure and paths.

### 2. Run the Training Script

To run the script and train the models, execute the following:

```bash
python train_models.py
```

This will automatically:
- Train each model (`1_4`, `5_8`, `9_12`, and `1_12`) separately.
- Use a learning rate scheduler and early stopping for efficient training.
- Save the trained models to disk.

### 3. Monitor Training

The training process includes detailed logs of loss, accuracy, and validation metrics for each model. Additionally, progress bars are displayed for each epoch.

### Example Output

```
Training model 1_4...
Epoch: 1/50 - Model: 1_4
Training Epoch 1: 100%|██████████| 113/113 [00:13<00:00,  8.41it/s, Loss=0.0138, Accuracy=0.8156]
Validation - Accuracy: 0.7075, Loss: 0.6792
...
Early stopping triggered.
1_4 model saved successfully!

Evaluating 1_4 model on test data...
Test - Accuracy: 0.8833, Loss: 0.2706
```

### 4. Evaluate the Model

The model is automatically evaluated on the test set after training. You can check the output for accuracy and loss on the test set.

## Configuration

The configuration options can be adjusted by editing the `Config` class in the script.

```python
class Config:
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    data_paths = {
        '1_12': r'/content/Final Testing Dataset',
        '1_4': r'/content/1-4',
        '5_8': r'/content/5-8',
        '9_12': r'/content/9-12',
        'Top': r'/content/Top_level'
    }

    # Training parameters
    training_params = {
        'batch_size': 32,
        'num_workers': 4,
        'epochs': 50
    }

    # Learning rates for different models
    learning_rates = {
        '1_12': 0.001,
        '1_4': 0.0001,
        '5_8': 0.0001,
        '9_12': 0.0001,
        'Top': 0.0001
    }

    # Number of classes for each model
    num_classes = {
        '1_12': 12,
        '1_4': 4,
        '5_8': 4,
        '9_12': 4,
        'Top': 3
    }

    # Model save paths
    model_save_paths = {
        '1_12': '1_12_bats.pth',
        '1_4': '1_4_model.pth',
        '5_8': '5_8_model.pth',
        '9_12': '9_12_model.pth',
        'Top': 'top_model.pth'
    }
```

You can modify paths, batch sizes, learning rates, and the number of epochs in this section.

## Future Enhancements

- Experiment with other neural network architectures like ResNet-50 or custom CNNs.
- Use advanced data augmentation techniques to improve model generalization.
- Implement a better model selection mechanism or ensemble models.

## Contact

For any questions or issues, please feel free to contact the project maintainer.

---

This `README.md` now includes the details from "The Enhancement and Automated Counting of Bats" project and the PyTorch-based image classification process. Let me know if you need any further updates!
