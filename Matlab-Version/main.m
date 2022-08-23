Folder_org_images     = 'org_data'; 
% Defining the folder for the original images

Folder_GT_images       = 'GT';       
% Defining the folder for the Ground Truth Images

Skin_images_Store 	   = imageDatastore(Folder_org_images,'IncludeSubfolders',true,'LabelSource','foldernames'); 
% Create datastore for large collections of data

Skin_Seg_images_Store    = imageDatastore(Folder_GT_images  ,'IncludeSubfolders',true,'LabelSource','foldernames'); 
% Create datastore for large collections of data

