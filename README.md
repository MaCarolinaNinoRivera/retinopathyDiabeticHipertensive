<p align="center"><img width="100%" src="ML/others/logo/torch_and_tf.svg" /></p>

--------------------------------------------------------------------------------
# Abstract
The shared code allows segmentation and classification of both diabetic and hypertensive retinopathy images.
It can be classified through the segmented images or the retina backgrounds. It also allows generating the masks
required by the UNET convolutional network.

# Dataset
- [Kaggle: Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
  Contains a total of 88,702 images divided into a total of 35,126 training images and 53,576 test images, all in "JPEG" format.
- [Drive](https://paperswithcode.com/dataset/drive)
  Contains 40 images, and 40 masks divided into 2 for training and validation, in "TIF" format.
- [Mendeley](https://data.mendeley.com/datasets/3csr652p9y/2)
  Contains 100 digital retinal fundus images and 100 masks taken at the Armed Forces Institute of Ophthalmology (AFIO), 
  Rawalpindi, Pakistan, for computer-aided diagnosis of hypertensive retinopathy, diabetic retinopathy, and papilledema
- [Odir](https://data.mendeley.com/datasets/3csr652p9y/2)
  An ophthalmic database of 5000 patients with left and right eye photographs, collected by Shanggong Medical Technology Co., 
  Ltd. from different hospitals/medical centers in China. 
  
# Code
- [Preprocessing]: Pre-processing of images to improve them and to increase the quantity
	* [data_aug.py]: File that allows you to create horizontal and vertical rotations and CLAHE enhancements to increase the initial data set
	* [preprocess_images.py]: This method allows you to reduce a thick frame of images, round them and resize them
	* [CalculateMeanStd.py]: This method allows calculating the average and standard values in order to carry out the normalization
- [Segmentation]: Image segmentation is the process of dividing an image into multiple parts or regions that belong to the same class. 
                  This task of clustering is based on specific criteria, for example, color or texture.
                  Read more at: https://viso.ai/deep-learning/image-segmentation-using-deep-learning/
    * [segmentacion.py]: This method allows to take an image and through morphology processes, change images to extract the blood vessels 
	* 	                 and the most important characteristics of the retina funds, this will be used as the masks in the UNet convolutional network.
	* [UNET]: The most important files are train and test must be executed in that order. After training the model, the Test function finally generates the masks.
- [Clasification]: Building your computer vision model is a sophisticated process that involves several steps, a high-level engineering team,
  and hundreds to thousands of images. Your model must be trained to identify these images through a process known as image classification 
  (or categorization as we refer to it in the Superb AI suite), which uses an advanced algorithm to assign a label or a tag to identify each individual image.
  Read more at: https://superb-ai.com/blog/an-introduction-to-image-classification-superb-ai-tutorial/
  * [train.py]: It allows a first classification, at the end it generates a csv file with the extracted characteristics and that will 
  * be sent to the second network when reading this file
  * [train_blend.py]: Read the file generated in the previous step and allow for better sorting