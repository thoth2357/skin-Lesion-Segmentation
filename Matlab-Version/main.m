Folder_org_images     = 'org_data'; 
% Defining the folder for the original images

Folder_GT_images       = 'GT';       
% Defining the folder for the Ground Truth Images

Skin_images_Store 	   = imageDatastore(Folder_org_images,'IncludeSubfolders',true,'LabelSource','foldernames'); 
% Create image datastore for storing images data

Skin_Seg_images_Store    = imageDatastore(Folder_GT_images  ,'IncludeSubfolders',true,'LabelSource','foldernames'); 
% Create datastore for storing images data

original_image_files    = Skin_images_Store.Files;    
% Get all Files Of Skin_images_Store datastore into variable 

ground_truth_image_files = Skin_Seg_images_Store.Files; 
% Get all Files Of Skin_Seg_images_Store datastore into variable

for i = 1:numel(original_image_files)           % Looping through to process all image
    Image_name    = original_image_files{i};    % Get org image By Indexing our Files from datastore  
    SegImage_name = ground_truth_image_files{i}; % Get org image By Indexing our Files from datastore 
    [original_image,segmented_image,DS_SCORE] =Segmentation_function(Image_name,SegImage_name,i);
    DS(i) = DS_SCORE;
end



