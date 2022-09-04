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
    [segmented_image,DS_SCORE] =Segmentation_function(Image_name,SegImage_name,i);
    DICE_SCORE(i) = DS_SCORE;
end

test_image_files=[];
segmented_image_files=[];

test_image_files{1} = 'ISIC_0000019.jpg';
test_image_files{2} = 'ISIC_0000095.jpg';
test_image_files{3} = 'ISIC_0000214.jpg';

segmented_image_files{1} = 'ISIC_0000019_Segmentation.png';
segmented_image_files{2} = 'ISIC_0000095_Segmentation.png';
segmented_image_files{3} = 'ISIC_0000214_Segmentation.png';

figure(1)
%%looping through test image files and getting the segmented lesion object with our function
for i = 1:numel(test_image_files)        
    test_image_name    = test_image_files{i};   
    segmented_image_name = segmented_image_files{i};
    [segmented_image,M,DS_SCORE] =Segmentation_function(Image_name,SegImage_name,i);
    
    %plotting
    subplot(2,3,i);
    imshow(test_image_name)
    subplot(2,3,i+3);
    imshow(M)
    
    X = ['DS :', num2str(DS_SCORE)];
    title(X) 
end
file_name = 'Segmentation-plot';
print('-dpng','-r600','-painters',file_name);


figure(2)
bar(DICE_SCORE)
xlabel('Image number');
ylabel('Corresponding DS value');
title('DS value for each of the Images')
file_name = 'DS_Values-plot';
print('-dpng','-r600','-painters',file_name);


mean_x = ['The mean DS value of all 60 images is : ', num2str(mean(DICE_SCORE))];
disp(mean_x)
mean_y = ['The standerd divation DS value of all 60 images is : ', num2str(std(DICE_SCORE))];
disp(mean_y)

%% Function
function [segmented_image,M,DS_SCORE] =Segmentation_function(Image_name,SegImage_name,i)
    segmented_image = im2double(imread(SegImage_name)); 
    %% Read the groud truth image and make it a double

    image = im2double(imread(Image_name)); 
    %% Read the orignal image and make it a double
    
    [m, n, k] = size(image);
    %% get a row vector whose elements are the lengths of the corresponding dimensions of image
    
    image = imresize(image,[800 1200]);
    %% resizing image to our desired 800x1200
  
    image= image(200:800-200,200:1200-200,:);
    %% extracting elements from resized image
    
    image = im2double(image);
    %% converting image array to double
    
    %%Converting image from RGB to Gray
    gray_image = rgb2lab(image);
    f = 0;
    gray_image_shaped = reshape(bsxfun(@times,cat(3,1-f,f/2,f/2),gray_image),[],3);
    [C,S] = pca(gray_image_shaped);
    S = reshape(S,size(gray_image));
    gray_Image_final = mat2gray(S(:,:,1));
    
    %% Morphological Closing
    se = strel('disk',1);
    close = imclose(gray_image_final,se);
    
    %% Complement Image
    K= imcomplement(close);
    
    %% 2-D wavelet Decomposition using B-Spline
    [cA,cH,cV,cD] = dwt2(K,'bior1.1');
    
    %% Otsu thresholding on each of the 4 wavelet outputs
    thresh1 = multithresh(cA);
    thresh2 = multithresh(cH);
    thresh3 = multithresh(cV);
    thresh4 = multithresh(cD);
    
    %% Calculating new threshold from sum of the 4 otsu thresholds and dividing by 2
    level = (thresh1 + thresh2 + thresh3 + thresh4)/2;
    
    %% single level inverse discrete 2-D wavelet transform

    X = idwt2(cA,cH,cV,cD,'bior1.1');
    
    %% Black and White segmentation
    BW=imquantize(X,level);
    
    %% Post Processsing
    BW = (BW - min(min(BW)))/(max(max(BW)) - min(min(BW)));
    BW = padarray(BW,[200 200],0,'both');
    SE = strel('disk',10);
    BW = imerode(BW,SE);
    SE = strel('disk',36);
    BW = imdilate(BW,SE);
    M  = imresize(BW, [m n]);

	%% Evaluation
    XX = ['processing image: ' num2str(i)];
    disp(XX)
    DS_SCORE = 2*(sum(sum(and(segmented_image,M))))/(sum(segmented_image(:))+ sum(M(:)));
end