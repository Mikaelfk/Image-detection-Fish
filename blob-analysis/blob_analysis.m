clear

myFolder = '/home/ukhu/Documents/fisk/testing-numbered';
filePattern = fullfile(myFolder, '*.jpg');
theFiles = dir(filePattern);

truth = readmatrix('Documents/fisk/GroundTruth-numbered-only.csv');

IoUScores = [];
bboxFish = zeros(length(theFiles), 4);
overlapTotal = 0;
bboxCounter = 0;

tic
for i = 1:height(truth)
    % Gets the bounding box from the ground truth
    bboxFish(truth(i,1), :) = [truth(i, 4), truth(i, 2), truth(i,5)-truth(i,4), truth(i,3)-truth(i,2)];
end

for k = 1:length(theFiles)
    
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    image = imread(fullFileName);

    % Find a binary image using thresholding
    [binaryImage, imageMasked] = createMask(image);

    % binaryImage = imfill(binaryImage, 'holes');

    % Remove disturbances from the binary image
    rectElem = strel('rectangle',[1,2]);
    Ibwopen = imopen(binaryImage, rectElem);

    hBlobAnalysis = vision.BlobAnalysis('MinimumBlobArea', 17500);
    [objArea, objCentroid, bboxOut] = step(hBlobAnalysis, Ibwopen);

    % If no bbox is found, try another mask
    if isempty(bboxOut)
         % Find a binary image using thresholding
        [binaryImage, imageMasked] = createMask2(image);

        % Remove disturbances from the binary image
        Ibwopen = imopen(binaryImage, rectElem);

        hBlobAnalysis = vision.BlobAnalysis('MinimumBlobArea', 1500);
        [objArea, objCentroid, bboxOut] = step(hBlobAnalysis, Ibwopen);
    end

    % If no bbox is found, try another mask
    if isempty(bboxOut)
        % Find a binary image using thresholding
        [binaryImage, imageMasked] = createMask3(image);

        % Remove disturbances from the binary image
        Ibwopen = imopen(binaryImage, rectElem);

        hBlobAnalysis = vision.BlobAnalysis('MinimumBlobArea', 2000);
        [objArea, objCentroid, bboxOut] = step(hBlobAnalysis, Ibwopen);
    end

    % If there still is not a bbox found, give up
    overlapRatio = 0;
    if isempty(bboxOut)
        disp("No bounding box found")
    else
        disp("Bounding box found")
        overlapRatioArray = bboxOverlapRatio(bboxOut, bboxFish(k, :));
        for i=1:length(overlapRatioArray)
            overlapRatio = overlapRatio + overlapRatioArray(i);
        end
        overlapRatio = overlapRatio / length(overlapRatioArray);
    end
    IoUScore(k) = overlapRatio;
    overlapTotal = overlapTotal + overlapRatio;
    bboxCounter = bboxCounter + 1;

    
    % Add rectangle to the image
    %Ishape = insertShape(image, 'Rectangle', bboxOut,'Linewidth',4, 'Color', 'red');


    %imwrite(Ishape, "/home/ukhu/Documents/fisk/output/"+baseFileName)

end
toc
% Blob detection
IoUScore(4, :) = IoUScore(1, :);
% Template matching
IoUScore(1, :) = [0    0.074712    0.067459           0     0.20054     0.55303           0     0.12463           0           0    0.037758           0     0.09151     0.19588     0.74586    0.049459           0           0     0.16134     0.78769     0.53805     0.87691           0           0     0.73667     0.89328           0           0     0.79997     0.89259     0.72269     0.74271     0.92929           0     0.72379     0.81923      0.2207     0.22665     0.15527     0.20499     0.37086           0    0.061788    0.091524     0.19644     0.08061           0      0.1486     0.21341     0.11256];
% SIFT
IoUScore(2, :) = [0.048267822665840954, 0.305043964403871, 0.03697602413464592, 0.3122107216214925, 0.22833140763804885, 0.6985469562557357, 0.706446898895497, 0.21020795840831832, 0.5014468616633705, 0.22224566214862085, 0.04512676056338028, 0.8127978033219372, 0.07227881914851021, 0.0857320540156361, 0.43396297647961946, 0.37452969739846287, 0.5731531131804926, 0.3547267671505985, 0.2141458531248766, 0.33158925995306077, 0.0, 0.22630704392173548, 0.37257976958803307, 0.31396454422770215, 0.06563031651367571, 0.30744657704049, 0.07701953666295841, 0.4798640061521026, 0.16606370385121755, 0.23087251755137533, 0.12930031161224925, 0.13861936152877566, 0.19580589435176257, 0.04600582080815243, 0.1505376723569001, 0.33272308923569427, 0.3679112341149437, 0.35397559090710257, 0.18549280177187155, 0.32835752300239907, 0.6881701529267007, 0.18837642531065119, 0.05368671423717295, 0.14192664465557128, 0.3420727401129944, 0.49698140545761893, 0.2557178585668201, 0.2018631069814803, 0.6730526846483489, 0.5000348334958896];
% SURF
IoUScore(3, :) = [0    0.087667    0.005267      0.1999      0.3445     0.20952     0.52659     0.24719     0.10773     0.75646     0.35061     0.32349     0.26312     0.10421     0.16422     0.27663      0.2852     0.30956     0.14454     0.36905     0.31846     0.53511     0.50339    0.036107      0.1028     0.30502     0.38321    0.035625           0     0.10639           0      0.3384     0.26671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
% YOLO v3
IoUScore(5, :) = [0.7055810667608619, 0.9612467676715086, 0.6337405770730439, 0.8798260146305878, 0.8717164937835162, 0.8839623723758173, 0.9381818181818182, 0.736282542195221, 0.910905795747543, 0.8895348837209303, 0.8198969813128894, 0.9565973594750573, 0.7782312925170068, 0, 0.8656857208462364, 0.9435479367915574, 0.8420456818858294, 0.8602736371705794, 0.8781763129589216, 0.9119792510941805, 0.8938746438746439, 0.8078792363045497, 0.7513766661520251, 0.8760031471282455, 0.8884113015284855, 0.873781623042383, 0.806911286826002, 0.9171333040238795, 0.885812386079437, 0.7768193791642993, 0.7291446626845006, 0.9050684492468294, 0.8417091192803708, 0.4833146239412185, 0.8758124045276431, 0.9124068841510403, 0.8638872034902199, 0.8743435858964741, 0.7916210295728368, 0.9028146574614976, 0.8913726339733001, 0.898874396998392, 0.7529301453352086, 0.8880360816584902, 0.8527248469091033, 0.9134058258591952, 0.8600682593856656, 0.9366526990386985, 0.8559312187754153, 0.8641832336099852];
IoUScore = IoUScore';

methods = ["Template Matching", "SIFT", "SURF", "Blob Detection",  "YOLOv3"];
boxplot(IoUScore, methods)
xlabel("Methods")
ylabel("IoU Score")
overlapAverage = (overlapTotal / bboxCounter)
