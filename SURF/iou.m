function score = iou
truth = readmatrix('/home/magnus/Fish_project/Ground_truth/GroundTruth-numbered-only.csv');
tmpltTruth = readmatrix('output/output.csv');

score = 0;
iouScores = [];

% loop through data in truth for template matching
for i = 1 : size(tmpltTruth, 1)
    % loop through ground truth
    for j = 1 : size(truth, 1)
        % if match, add to IOU score
        if tmpltTruth(i, 1) == truth(j, 1)
            % if the position is -1, it didn't meet the threshold so the
            % score is just set to 0
            if (tmpltTruth(i,2) == -1 && ...
                tmpltTruth(i,3) == -1 && ...
                tmpltTruth(i,4) == -1 && ...
                tmpltTruth(i,5) == -1) 
                iouScores = [iouScores, 0];
                break
            end
            disp(i)
            
            % box from ground truth
            boxTruth = [truth(j, 4) truth(j, 2) truth(j, 5)-truth(j, 4) truth(j, 3)-truth(j, 2)];
            % box from template matching truth
            boxTmplt = [tmpltTruth(i, 4) tmpltTruth(i, 2) tmpltTruth(i, 5)-tmpltTruth(i, 4) tmpltTruth(i ,3)-tmpltTruth(i, 2)];
            
            % save overlap score
            iouScores = [iouScores, bboxOverlapRatio(boxTruth, boxTmplt)];
            break
        end
    end
end

disp(['[' num2str(iouScores(:).') ']']);

% returnaverage (a score of 1 is perfect)
score = sum(iouScores)/50;