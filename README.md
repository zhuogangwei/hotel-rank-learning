# Seeing Stars: A Computer Vision-Centered Approach for Star Rating in the Hospitality Industry
With over 10 million hotels globally, there is a need for travel booking websites to provide accurate
and reliable information to visitors. Star rating is most frequently used as a filtering criterion, but is
unreliable given the absence of commonly accepted standards for star rating assignment. Manual
human verification can be subjective and incurs high operating costs. Several major third-party
distribution platforms, e.g., Booking.com, therefore let hotel owners self-report their own star ratings,
with highly inconsistent results.

Our objective is to create a computer-vision-assisted machine learning model that can more accurately
assign hotel star ratings using images and meta-data (e.g. pricing, facilities). This promises a cheaper
and more objective methodology in assigning hotel star ratings.

The full project proposal can be found [here](https://github.com/ishakbhatt/hotel-rank-learning/blob/main/project_proposal/CS_230_Project_Proposal__Ye__Zhuo__Bhatt_.pdf).

## Training (MILESTONE ONLY)
### ResNet50-based CNN 
This model classifies hotels into their respective ratings using **exterior image data**.    

The model can be trained directly from `train_resnet50.py` from the `src/models` directory (do not train from root).   

When the program starts, you will be prompted with the following questions:    

`Would you like to download images from AWS S3? Y/N:`    
`Have you downloaded images already from exterior.zip? Y/N:`    

Answer `N` to both. A data sample is provided in the repository in `exterior.zip` that will be used for training.    

### Hotel Metadata DNN
This model classifies hotels into their respective ratings using **hotel metadata**.    

The model can be trained directly from `train_structured.py` from the `src/models` directory (do not train from root).

## Class distribution
Run `class_dist.py` from `src/preprocessing`. The class distribution (without augmentation) can be found in `data/data_analysis`.

## Libraries
`pillow`    
`numpy`    
`tensorflow`    
`shutil`    
`boto3`   
`zipfile`  
`os`  
`matplotlib`  
`sklearn`
`pandas`
