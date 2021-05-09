
%load the main image and make it a greyscale image
Test1 = imread('Cropped_fish1.jpg');
SampleFish1 = rgb2gray(Test1);

%find the files to loop  through and the output file and the ground truth
myFolder = '/home/magnus/Fish_project/Bildesett-testing/testing-numbered';
fileID = fopen('output/output.csv', 'w');
truth = readmatrix('/home/magnus/Fish_project/Ground_truth/GroundTruth-numbered-only.csv');

for i = 1:height(truth)
         % Gets the bounding box from the ground truth
            bboxFish(truth(i,1), :) = [truth(i, 4), truth(i, 2), truth(i,5)-truth(i,4), truth(i,3)-truth(i,2)];
end


filePattern = fullfile(myFolder, '*.jpg');
theFiles = dir(filePattern);

tic

%For loop for comparing the SURF key points between two fish
for k = 1 : length(theFiles)
    try 
        % load image
        [folder, baseFileName, extension] = fileparts(fullfile(theFiles(k).folder, theFiles(k).name));
        image = rgb2gray(imread(fullfile(folder, append(baseFileName,extension))));
        
    
        %Sample 1 is defined outside the loop
        %Use the current image from the file as sample two
        SampleFish2 = image;

        %Detects the key points in both images
        points1 = detectSURFFeatures(SampleFish1, 'MetricThreshold',50,'NumScaleLevels',5,'NumOctaves', 4);
        points2 = detectSURFFeatures(SampleFish2, 'MetricThreshold',50,'NumScaleLevels',5,'NumOctaves', 4);
    
        %Find the 300 strongest key points in both the images
        strongest1 = points1.selectStrongest(300);
        strongest2 = points2.selectStrongest(300);

        %matching the features with eachother
        [fishFeatures1, points1] = extractFeatures(SampleFish1, points1);
        [fishFeatures2, points2] = extractFeatures(SampleFish2, points2);

        fishPairs = matchFeatures(fishFeatures1, fishFeatures2);

        matchedFishPoints1 = points1(fishPairs(:, 1), :);
        matchedFishPoints2 = points2(fishPairs(:, 2), :);
   
        % Line 51 - 74 are commented out. This is for the first SURF
        % implementation and draws the blue plygons. This is not necesarry
        % for the final product.
        % Geometric transformation to eliminate the outliers
        
        %[tform, inlierIdx] = ...
        %estimateGeometricTransform2D(matchedFishPoints1, matchedFishPoints2, 'affine');
        %inlierFishPoints1 = matchedFishPoints1(inlierIdx, :);
        %inlierFishPoints2 = matchedFishPoints2(inlierIdx, :);
    
        %creating a boxpolygon for the box around the fish
        
        %boxPolygon = [1, 1;...                        
        %    size(SampleFish1, 2), 1;...                 
        %    size(SampleFish1, 2), size(SampleFish1, 1);... 
        %    1, size(SampleFish1, 1);...                 
        %    1, 1];
    
        % transformPointsForward is a forward geometric transformation based
        % on the coordinates we get from the inlier key points
        
        %newBoxPolygon = transformPointsForward(tform, boxPolygon);
    
        % Draw the polygon around the fish
        
        %pgon = polyshape(newBoxPolygon);
    
        % Finding the coordinates of the SURF points of the fish in FishPoints2
        location = matchedFishPoints2.Location;
    
        % Setting the values of X and Y from the coordinates of the SURF
        % points
        LargestX = location(1);
        LargestY = location(1, 2);
        SmallestX = location(1);
        SmallestY = location(1, 2);
        
        % Looping through every SURF point finding the largest and smallest
        % X and Y values
        for i = 1 : matchedFishPoints2.size
        
            if(location(i) > LargestX)
                 LargestX = location(i);
            end
        
            if(location(i) < SmallestX)
                SmallestX = location(i);
            end
        
            if(location(i, 2) > LargestY)
                 LargestY = location(i, 2);
            end
        
            if(location(i, 2) < SmallestY)
                 SmallestY = location(i, 2);
            end
        end
        
        % Finding the distance between the different X and Y
        % coordinates
        DistanceX = LargestX - SmallestX;
        DistanceY = LargestY - SmallestY;
        
        % The position and size of the ground truth and the bounding box
        rectanglePosAndSize = [SmallestX SmallestY DistanceX DistanceY];
        groundTruthPosAndSize = bboxFish(k, :);
        
        % Drawing the figure and the bounding boxes
        figure('visible', 'off');
        imshow(SampleFish2);
        rectangle('position', rectanglePosAndSize, 'edgecolor', 'r', 'linewidth',2);
        rectangle('position', groundTruthPosAndSize, 'edgecolor', 'g', 'linewidth', 2)
        hold on;
        %plot(pgon)
        title('Detected Box');
        saveas(gcf,"./output/" + baseFileName + extension);
        disp(rectanglePosAndSize)
    
        fprintf(fileID, ...
                '%s,%d,%d,%d,%d\n', ...
                baseFileName, ...
                round(rectanglePosAndSize(2)), ... % y1
                round(rectanglePosAndSize(2)+rectanglePosAndSize(4)), ... %y2
                round(rectanglePosAndSize(1)), ... % x1
                round(rectanglePosAndSize(1)+rectanglePosAndSize(3)) ... % x2
            );
    
    
        catch e
            disp("wopsie poopsie")
            disp(baseFileName);
            disp(e.message)
        
        end

end


toc
fclose(fileID);


%disp("iou: " + iou)
