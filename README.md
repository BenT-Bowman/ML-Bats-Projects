# The Enhancement and Automated Counting of Bats

## Overview

This project focuses on the ecological importance of bats by enhancing the accuracy of bat counting in small, low-resolution images using convolutional neural networks (CNNs). Bats play a crucial role in ecosystems, contributing to pollination, seed dispersal, and insect population control. Accurate population assessments of bats are essential for ecological research and conservation strategies.

Our study aims to overcome the challenges posed by low-quality images, such as blurriness, background clutter, and overlapping bats, which traditionally lead to inaccurate counts and misinterpretations. By leveraging advanced machine learning techniques, we have developed a robust and accurate model to detect and count bats in these challenging images.

## Features

- **Convolutional Neural Networks (CNNs):** The project utilizes a two-tiered CNN architecture to detect and count bats. The model is capable of handling images with varying numbers of bats, ranging from 1 to 12.
  
- **Synthetic Data Augmentation:** To address class imbalances and improve the model's accuracy, synthetic images were generated to simulate a variety of realistic scenarios.

- **Ensemble Methodology:** The ensemble model enhances the overall accuracy by using sub-models to focus on different subsets of the bat count problem.

- **Real-world Testing:** The model was validated through real-world applications, where automated predictions were compared against manual counts by professionals.

## Project Structure

- `Final_Project_Training_1_1.ipynb`: The main notebook where the model is trained, validated, and tested. It includes the data preprocessing, model training, and performance evaluation.
- `random_placement_2.ipynb`: A supporting notebook that deals with the synthetic data generation and random placement of bats in images.
- `The Enhancement and Automated Counting of Bats.pdf`: The project report that details the methodology, results, and discussion of the study.

