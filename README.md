### Dataset Acquisition and Exploratory Data Analysis (EDA)

The datasets for both grading (classification) and segmentation tasks were acquired from IEEE Dataport.

#### Segmentation Task

The segmentation task involves identifying and segmenting five distinct classes within retinal images:

1. Microaneurysms
2. Haemorrhages
3. Soft Exudates
4. Hard Exudates
5. Optic Disk

The dataset contains 54 images designated for training purposes. For each image, binary ground truth masks were provided corresponding to the above classes. Images and corresponding masks have a resolution of 4288 by 2848 pixels.

#### Observations from Initial EDA:

* Some ground truth masks were missing:

  * Haemorrhage class: Missing mask for 1 image.
  * Soft exudates class: Missing masks for 28 images.

Based on visual inspection and domain knowledge, an educated assumption was made that the absence of these masks indicates the absence of these lesion types in the respective images.

#### Pixel-Level Representation:

* Microaneurysms: Approximately 0.1%
* Haemorrhages: Approximately 1%
* Hard Exudates: Approximately 0.8%
* Soft Exudates: Approximately 0.4%
* Optic Disk: Approximately 1.8%
* Background: Remaining percentage

#### Disease Grading Dataset

The disease grading dataset contains 413 training images and an accompanying CSV file with the following annotations:

* **Retinopathy Grade (0-4):**

  * Grade 0: 32%
  * Grade 1: 5%
  * Grade 2: 33%
  * Grade 3: 18%
  * Grade 4: 12%

* **Risk of Macular Edema (0-2):**

  * Risk 0: 43%
  * Risk 1: 10%
  * Risk 2: 47%

### Experimental Setup

A highly configurable basic setup was established for the segmentation task, allowing flexibility in experimenting with different parameters. Configurations, such as loss functions, augmentations, and preprocessing methods, could be easily switched, facilitating rapid experimentation and quick iterations.

#### Planned Experimental Variations

The following experimental ideas were considered:

* Augmentation strategies:

  * No augmentation
  * Light geometric augmentation
  * Color and light geometric augmentation
  * Comprehensive augmentations

* Choice of neural networks

* Selection of segmentation models

* Evaluation of various loss functions

* Experimentation with different image sizes

* Application of diverse preprocessing techniques

Due to time and computational resource limitations, experiments were confined to these specific configurations.

### Final Experimental Results

For the segmentation task, experiments were conducted specifically on two classes: haemorrhages and hard exudates. The final optimal configuration identified was:

* Comprehensive augmentations
* Neural Network: "resnext50\_32x4d"
* Segmentation Model: SegFormer
* Loss Functions: Combination of Focal Dice and Tversky losses
* Image Size: 1024 x 1024 pixels
* Preprocessing: Standardization method inspired by Ben Graham’s research, resizing, and normalization

Some of these findings were directly applied to the classification and multi-task setups. For the classification task, the optimal combined loss was identified as a mix of focal loss and cross-entropy loss.

### Network Architectures and Configurations

#### Mixture of Experts Model

Implemented with a ResNeXt50 backbone, this model dynamically selects specialized experts via a router mechanism. It integrates an FPN decoder for segmentation tasks and a dedicated classification head. Configurations included comprehensive augmentations, combined segmentation losses (Focal Dice and Tversky), advanced fundus preprocessing, and detailed hyperparameters optimized through experimentation.

#### Multi-Head Model

Built using SegFormer architecture for segmentation, with an additional classification head added on top of the encoder. This model aimed to simultaneously perform segmentation and classification, leveraging optimal parameters from individual task models.

#### Classification Model

Focused solely on disease grading, using a ResNeXt50 backbone. The best loss function combination found was focal loss with cross-entropy. Similar augmentation and preprocessing techniques used for segmentation tasks were also applied here to ensure consistency.

#### Segmentation Model

Standalone segmentation model employing the SegFormer architecture with ResNeXt50 backbone. Comprehensive augmentations, advanced preprocessing methods, and combined loss functions were used to achieve the best results on haemorrhages and hard exudates classes.

### Performance Metrics Summary

* **Classification (separate)**: Highest accuracy achieved was **71%**.
* **Segmentation (separate)**: Highest mean Dice score achieved was **0.7626**.
* **Multihead (combined)**: Best combined model yielded a classification accuracy of **66%** and a mean Dice score of **0.59**.
* **Mixture of Experts (combined)**: Best combined model yielded a classification accuracy of **64%** and a mean Dice score of **0.75**.

For detailed metrics, refer to the [WandB Project](https://wandb.ai/vinay-rp-36-personal/final_retinopathy_multitask).

Due to time and computational constraints, more extensive experimentation with these models, particularly the Multi-head and Mixture of Experts architectures, remains for future work.
