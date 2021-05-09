
%load the main image and make it an rgb
Test1 = imread('Cropped_fish1.jpg');
SampleFish1 = rgb2gray(Test1);

%find the files to loop  through
myFolder = '/home/magnus/Fish_project/Bildesett-testing/testing-numbered';
filePattern = fullfile(myFolder, '*.jpg');
theFiles = dir(filePattern);

%For loop for comparing the SURF key points between two fish. This script
%is only for displaying the SURF points on the fish and the matching key
%points
for k = 2  %: length(theFiles)
    % load image
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    image = rgb2gray(imread(fullFileName));
    
    %Sample 1 is defined outside the loop
    %Use the current image from the file as sample two
    SampleFish2 = image;

    %Detects the key points in both images
    points1 = detectSURFFeatures(SampleFish1, 'MetricThreshold',50,'NumScaleLevels',5, 'NumOctaves', 4);
    points2 = detectSURFFeatures(SampleFish2, 'MetricThreshold',50,'NumScaleLevels',5, 'NumOctaves', 4);
    
    %Find the 300 strongest key points in both the images
    strongest1 = points1.selectStrongest(300);
    strongest2 = points2.selectStrongest(300);

    %display the image with key points
    figure
    imshow(SampleFish1);    
    hold on; plot(strongest1);

    %display the image with key points
    figure
    imshow(SampleFish2); 
    hold on; plot(strongest2);


    %matching the features with eachouther
    [fishFeatures1, points1] = extractFeatures(SampleFish1, points1);
    [fishFeatures2, points2] = extractFeatures(SampleFish2, points2);

    fishPairs = matchFeatures(fishFeatures1, fishFeatures2);

    %display both images with lines between the matching key points
    matchedFishPoints1 = points1(fishPairs(:, 1), :);
    matchedFishPoints2 = points2(fishPairs(:, 2), :);
    figure;
    showMatchedFeatures(SampleFish1, SampleFish2, matchedFishPoints1, ...
    matchedFishPoints2, 'montage');
    title('Putatively Matched Points (Including Outliers)');
    
    [tform, inlierIdx] = ...
    estimateGeometricTransform2D(matchedFishPoints1, matchedFishPoints2, 'affine');
    inlierFishPoints1 = matchedFishPoints1(inlierIdx, :);
    inlierFishPoints2 = matchedFishPoints2(inlierIdx, :);
    
    figure;
    showMatchedFeatures(SampleFish1, SampleFish2, inlierFishPoints1, ...
    inlierFishPoints2, 'montage');
    title('Matched Points (Inliers Only)');
    
    
end
