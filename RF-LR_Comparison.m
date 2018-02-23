%% Comparison of Random Forests and Logistic Regression on Wisconsin Breast Cancer Dataset (Diagnostic)
 
%% Initialisation
clear ; close all; clc
format compact; %Suppress the display of blank lines
rng default;    %Ensure repeatable results
 
%% Load data
%1) ID number
%2) Diagnosis (M = malignant, B = benign)
%3-32) Ten real-valued features are computed for each cell nucleus. The mean, standard error, and...
%      "worst" values of these features were computed for each image, resulting in 30 features.
 
T = readtable('wdbc.data.csv');
T.Properties.VariableNames = {'ID' 'Diagnosis' 'RadiusMean' 'TextureMean' 'PerimeterMean' 'AreaMean'...
    'SmoothnessMean' 'CompactnessMean' 'ConcavityMean' 'ConcavePointsMean' 'SymmetryMean' 'FractalDimensionMean'...
    'RadiusSE' 'TextureSE' 'PerimeterSE' 'AreaSE' 'SmoothnessSE' 'CompactnessSE' 'ConcavitySE' 'ConcavePointsSE'...
    'SymmetrySE' 'FractalDimensionSE' 'RadiusWorst' 'TextureWorst' 'PerimeterWorst' 'AreaWorst' 'SmoothnessWorst'...
    'CompactnessWorst' 'ConcavityWorst' 'ConcavePointsWorst' 'SymmetryWorst' 'FractalDimensionWorst'};
 
head(T,5) % view first 5 rows
T.ID =[]; % drop ID column
% confirm there are no missing values
nMissing = sum(sum(ismissing(T)));
fprintf('Number of missing values = %i\n', nMissing)
 
%% Visual exploration of data and preparation for classification task
% convert Diagnosis to categorical array and check class imbalance
T.Diagnosis = categorical(T.Diagnosis);
summary(T.Diagnosis)
histogram(T.Diagnosis); title('Class imbalance');
fprintf('Major class represents %.2f%% of all cases\n', 100* length(T.Diagnosis(T.Diagnosis == 'B')) / height(T))
 
% replace Diagnosis column with dummy variables in order to perform logistic regression
dummyDiagnosis=zeros(height(T),1);
dummyDiagnosis(T.Diagnosis=='M') = 1;
T.Diagnosis=[]; T.Diagnosis = dummyDiagnosis; %replace categorical column with dummy variable column
% NB if we had not previously converted to a categorical array then we could have used:
% T.Diagnosis(strcmp(T.Diagnosis,'B')) = {0}; % and similarly 'M' = {1}
 
% Split table into separate arrays of attributes and labels
X = table2array(T(:,1:end-1));
y = table2array(T(:,end));
 
% Perform LDA and view decision boundary
% normalise data
Z = zscore(X);
% Perform LDA
MdlLinear = fitcdiscr(Z,y);
% Get coefficients for linear boundary
K = MdlLinear.Coeffs(1,2).Const;
L = MdlLinear.Coeffs(1,2).Linear;
% Project the data onto the linear discriminant eigenvector
xL = Z*L + K;
% Plot histogram of data on  LDA space
figure; hold on;
histogram(xL(y==0), 'binWidth', 2) %,'Normalization','pdf')
histogram(xL(y==1), 'binWidth', 2) %,'Normalization','pdf')
line([K,K],[0,90], 'Color','k', 'LineStyle', '--', 'LineWidth', 0.75)
legend({'Benign','Malignant','Decision boundary'}, 'Location', 'northwest')
xlabel('LD1'); ylabel('Observation counts'); title('LDA of Wisconsin Breast Cancer Dataset')
 
% View correlations
R = corrcoef(X);
figure;
heatmap(R, 'Colormap', flipud(autumn)); title('Correlation matrix (30 predictors)')
 
% Calculate skew
sk=skewness(X);
skMax = max(sk); isk = find(sk==skMax);
fprintf('Predictor %s has maximum skewness = %f \n', char(T.Properties.VariableNames(isk)), skMax)
 
%{
%% Extension: Use PCA to transform factors
% I experimented with this for the RF model and there was no difference in test performance
% with either all factors or 17 factors (99% R^2)
% Standardise variables and perform PCA
Z = zscore(X);
%X = Z(:,1:17);
[coeff,score,latent,~,explained,~] = pca(Z);
% Plot explained variance of first 10 factors
bar(explained(1:30)); title('Explained variance for PCA factors'); ylabel('R squared'); xlabel('Factor #');
h=gca; h.XTick = 1:30;
%}
 
% Future work: calculate Bayesian error rate using LDA decision boundary
 
%% Create training and test sets, and cross validation folds
% Split data into training set (70%) and test set (30%) using stratified sampling
% Note this does not prevent the introduction of statistical bias
cv = cvpartition(y,'HoldOut',0.3);
X_train = X(cv.training,:); X_test = X(cv.test,:);
y_train = y(cv.training); y_test = y(cv.test);
 
% Split data for 10-fold cross-validation using stratified sampling
cv_train = cvpartition(y_train, 'KFold', 10);
 
%% Logistic Regression
 
% Optimise lambda hyperparameter and view results using Bayesian optimisation
% Optimise without cross validation. Result: lambda is very close to zero - the model is overfitting the data
% CHECK THIS STILL HOLDS WITH RANDOMISATION
optOptions1 = struct('Optimizer','bayesopt')
LR = fitclinear(X_train',y_train,'Learner','logistic','Regularization','ridge','ObservationsIn','columns',...
                'OptimizeHyperparameters',{'Lambda'},'HyperparameterOptimizationOptions',optOptions1);
title('Objective function model: no cross-val')
 
% Optimise with 10-fold cross validation. Result: Larger optimal value of lambda (however still small)
optOptions2 = struct('Optimizer','bayesopt','MaxObjectiveEvaluations',50,'CVPartition',cv_train)
[LR,FitInfo,OptResults] = fitclinear(X_train',y_train,'Learner','logistic','Regularization','ridge',...
                          'ObservationsIn','columns','OptimizeHyperparameters',{'Lambda'},...
                          'HyperparameterOptimizationOptions',optOptions2);
title('Objective function model: 10-fold cross-val')
% Use the optimised value of lambda for our best model
lambdaBest = table2array(OptResults.XAtMinEstimatedObjective); % Lambda = 0.0422
 
%{
% Let's cheat and use the test data to check the optimisation. Result: Much larger optimal value of lambda
% Suggests still significant overfitting in 10-fold CV? Not sure why CV does not better address the overfitting?
% Having completed the experiment there does not appear to be an issue with small lambda values
vd = {X_test',y_test}
LR = fitclinear(X_train',y_train,'Learner','logistic','Regularization','ridge','ObservationsIn','columns',...
                'OptimizeHyperparameters',{'Lambda'},'HyperparameterOptimizationOptions',optOptions1,'ValidationData',vd);
title('Objective function model: peaking at validation data')
%}
 
% Test the above by plotting training and test performance for a range of lambda values using cross-validation
% Calculate training set performance
lambda = [0,10.^[-5:1:2]];
cvLR = fitclinear(X_train',y_train,'Learner','logistic','Regularization','ridge','ObservationsIn','columns',...
                  'Lambda',lambda,'CVPartition',cv_train);
% r has dimensions (numFolds, numLambdas)
r= kfoldLoss(cvLR,'mode','individual');
trainErrorsLR = kfoldLoss(cvLR);
trainErrorsLRmin = min(r,[],1); trainErrorsLRmax = max(r,[],1);
 
% find model with lowest mean training error
idxBest = find(trainErrorsLR == min(trainErrorsLR(:)));
% the cross-val error is approximately flat and minima are found for the 2nd and 6th lambda values
format shortE;
lambda([2,6]) % Minima for lambda values of 1e-05 and 0.1
format short;
 
% Calculate test set performance
%Create matrix for test results. Note r has dimensions (numFolds, numLambdas)
r = zeros(cv_train.NumTestSets, numel(lambda)); % misclassification error
 
for i = 1:cv_train.NumTestSets
    Mdl = cvLR.Trained{i};
    [yPred, scores] = predict(Mdl,X_test);
    r(i,:) = loss(Mdl,X_test,y_test);
end
testErrorsLR = mean(r,1); testErrorsLRmin = min(r,[],1); testErrorsLRmax = max(r,[],1);
 
% Plot semi-log graph including error bars
figure;
yneg=trainErrorsLR - trainErrorsLRmin; ypos = trainErrorsLRmax - trainErrorsLR;
errorbar(lambda,trainErrorsLR,yneg,ypos,'color','r','marker','o'); hold on;
yneg=testErrorsLR - testErrorsLRmin; ypos = testErrorsLRmax - testErrorsLR;
errorbar(lambda,testErrorsLR,yneg,ypos,'color','b','marker','x');
ax=gca; ax.XScale='log'; ax.XLim=[0,1000]; ax.YLim=[0,0.16];
legend('Training error','Test error'); title('Training and test error as function of lambda')
xlabel('Lambda'); ylabel('Misclassification Error');
 
% The graph shows that the optimised value of lambda is reasonable
% Calculate performance for our best model
cvLRBest = fitclinear(X_train',y_train,'Learner','logistic','Regularization','ridge','ObservationsIn','columns',...
                  'Lambda',lambdaBest,'CVPartition',cv_train);
% Create container for best model results
numFolds = cv_train.NumTestSets;
bestLR.results = cell(numFolds,1);
% Calculate training errors. r has dimensions (numFolds,1)
r = kfoldLoss(cvLRBest,'mode','individual');
trE = kfoldLoss(cvLRBest); trEmin = min(r); trEmax = max(r);
bestLR.trainError = trE; bestLR.trainErrorMin = trEmin; bestLR.trainErrorMax = trEmax;
 
% perform predictions for best value of lambda
for i = 1:numFolds
    Mdl = cvLRBest.Trained{i}
    [yPred, scores] = predict(Mdl,X_test);
    bestLR.results{i}.MP = ModelPerformanceCalculator(y_test,yPred);
    bestLR.results{i}.scores = scores;
    r(i) = loss(Mdl,X_test,y_test);
end
teE = mean(r); teEmin = min(r); teEmax = max(r);
bestLR.testError = teE; bestLR.testErrorMin = teEmin; bestLR.testErrorMax = teEmax;
 
% Calculate model performance measures for fold with median performance
rs=sort(r); rMedian=rs(round(numel(rs)/2));
bestLR.idxBest = find(r==rMedian);
% Output performance measures for best LR model
fprintf('Best model performance for Logistic Regression:')
bestLR.results{bestLR.idxBest(1),1}.MP
 
% Optional: Look at coefficients to examine co-linearity in model, number of non-zero betas
% Optional: Re-run using feature selection
 
%% Random Forest
% Create hyperparameter search grid
% Search over numFeatures from 1 to a third of predictors
numFeatures = 1:round(size(X_train,2)/3); %10 features
numFeatRuns = length(numFeatures);
% Search over numTrees from 1 to half number of training samples
length(X_train)/2; % approx. 200
maxTrees = 200;
% For training we can quickly return the cumulative error for each tree, however for this is slower
% for test so we select a smaller range of 30 tree values to test: 1-10, 15 and 10-200 in steps of 10
predTrees = [1:10,15,20:10:200]; predTreeRuns = length(predTrees);
% Create results matrices
trainErrors = zeros(numFeatRuns,maxTrees);
oobErrors = zeros(numFeatRuns,maxTrees);
oobErrorsInd = zeros(numFeatRuns,maxTrees); %Individual errors for each tree in ensemble
testErrors = zeros(numFeatRuns,predTreeRuns);
% Run model over grid and store results
tic
for nF = 1:numFeatRuns
    % Train model
    % We are using OOB error to calculate an unbiased estimate of generalisable performance
    % Cross-validation may provide a better estimate by eliminating positive bias in the OOB error,
    % however it means the RF does not get to train on all of the data and also increases execution time.
    % 
    % Leave minLeafSize as 1 to grow deep trees as this maximises the model strength (i.e. minimises bias)
    
    % Because there are many co-linear features I experimented using curvature test as split criterion
    % instead of CART to minimise error, however this did not have a noticable effect
    
    % PredictorImportance could be used to select a subset of features to train on,
    % however note this can be problematic with correlated features
    
    predictorType = 'allsplits'; %'curvature';
    RF = TreeBagger(maxTrees, X_train, y_train,'Method','classification','OOBPrediction','on',...
         'NumPredictorsToSample',numFeatures(nF),'PredictorSelection',predictorType);
         %'OOBPredictorImportance','On','PredictorNames',T.Properties.VariableNames(1:30))
    fprintf('Trained Random Forest with %i features ',nF)
    toc
    % NumTrees is a 'free hyperparameter' i.e. we do not have to train the model separately 
    trainErrors(nF,:) = error(RF,X_train, y_train,'Mode','cumulative');
    oobErrors(nF,:) = oobError(RF,'Mode','cumulative');
    oobErrorsInd(nF,:) = oobError(RF,'Mode','individual');
    % Calculate predictions on test data
    
    for nT = 1:predTreeRuns
        % Perform prediction for sample values of numTrees up to maxTrees
        yPred = predict(RF,X_test,'trees',[1:predTrees(nT)]);
        testErrors(nF,nT) = sum(y_test~=str2num(cell2mat(yPred)))/length(y_test); %need to convert from cell array
    end
    fprintf('Calculated predictions with %i features ',nF)
    toc
end
 
% Examine the distribution of OOB errors as a function of numFeatures
% Through repeated runs there does not appear to be a clear minimum however 1 or 2 features tends to perform worse
% and the error appears to be a minimum around 5 features and beyond
% Note the default value is sqrt(num predictors) =~ 5
% Select the smallest numFeatures with best median OOB error
figure; boxplot(oobErrorsInd');
title('Training set OOB error as function of numFeatures');
xlabel('Number of features'); ylabel('Out-of-bag misclassification error');
medErr = median(oobErrorsInd'); idx = find(medErr == min(medErr));
numFeaturesBest= idx(1);
fprintf('Best number of features to select is %i\n', numFeaturesBest)
 
% Plot the out-of-bag error over the number of grown classification trees for numFeaturesBest
% OOB errors reach a minimum after approx. 100 trees. Therefore select this value for the best model
RF = TreeBagger(maxTrees, X_train, y_train,'Method','classification','OOBPrediction','on',...
     'NumPredictorsToSample',numFeaturesBest,'PredictorSelection',predictorType);
oobErr = oobError(RF);
figure; plot(oobErr); title(['Training set OOB error as function of number of trees (numFeatures=',...
                            num2str(numFeaturesBest), ')']);
xlabel('Number of grown trees'); ylabel('Out-of-bag misclassification error');
numTreesBest=100;
 
% Plot error as function of numFeatures for test set
% Through repeated runs the best number of features fluctuates but there is not a clear pattern
% It suggests using the default value of sqrt(num % predictors)
figure; plot(testErrors(:,end)); title('Test set error as function of numFeatures');
xlabel('Number of features'); ylabel('Misclassification error');
 
% Plot error surface. Training error surface becomes relatively flat after a very small number of trees
% We do not see overfitting at high numbers of trees
figure; view(3); hold on
[Xplot,Yplot] = meshgrid(1:maxTrees,numFeatures);
hSurface = surf(Xplot,Yplot,trainErrors); set(hSurface,'FaceColor',[1 0 0],'FaceAlpha',0.25,'EdgeColor',[0.3 0.3 0.3],'EdgeAlpha',0.25);
hSurface = surf(Xplot,Yplot,oobErrors); set(hSurface,'FaceColor',[0 0 1],'FaceAlpha',0.25,'EdgeColor',[0.3 0.3 0.3],'EdgeAlpha',0.25);
[Xplot,Yplot] = meshgrid(predTrees,numFeatures);
hSurface = surf(Xplot,Yplot,testErrors); set(hSurface,'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor',[0.3 0.3 0.3],'EdgeAlpha',0.25);
legend('Training error', 'OOB error', 'Test error'); 
hold off
 
% Perform cross-validation on the best RF model to allow a fair comparison with LR
% Create indices for 10 fold CV training sets
numFolds = cv_train.NumTestSets;
% Create cell array to hold training patitions as these could be of different size
testCV = cell(numFolds,2);
% Create container for model results
bestRF.results = cell(numFolds,1);
r = zeros(1,numFolds);
 
idx = zeros(size(X_train,1), numFolds);
for i = 1:numFolds
    idx(:,i) = training(cv_train,i);
    testCV{i,1} = X_train(idx(:,i)==1,:); %X_trainCV
    testCV{i,2} = y_train(idx(:,i)==1,:); %y_trainCV
    % Train model
    RFBest = TreeBagger(numTreesBest, testCV{i,1}, testCV{i,2},'Method','classification',...
                    'NumPredictorsToSample',numFeaturesBest);
    % Predict test set
    [yPred, scores] = predict(RFBest,X_test);
    bestRF.results{i}.MP = ModelPerformanceCalculator(y_test,str2num(cell2mat(yPred))); %need to convert from cell array
    bestRF.results{i}.scores = scores;
    r(i) = bestRF.results{i}.MP.MCE; % Misclassification error
end
teE = mean(r); teEmin = min(r); teEmax = max(r);
bestRF.testError = teE; bestRF.testErrorMin = teEmin; bestRF.testErrorMax = teEmax;
 
% Find model results for fold with median performance
rs=sort(r); rMedian=rs(round(numel(rs)/2));
bestRF.idxBest = find(r==rMedian);
% Output performance measures for best LR model
fprintf('Best model performance for Random Forest:')
bestRF.results{bestRF.idxBest(1),1}.MP
 
% Optional: reduce number of features using correlations / PCA / feature importance and see if can improve model
% I experimented using PCA on all factors and 17 factors (99% R^2) and there was no discernable difference
% Feature importance did indicate some of the more important predictors however there was significant variation
% between models and this has not been explored further
 
%{
% Optional - examine eature importance. Note this can be problematic with correlated features
imp = RF.OOBPermutedPredictorDeltaError;
figure; bar(imp); title('Curvature Test');
ylabel('Predictor importance estimates'); xlabel('Predictors');
h = gca;
h.XTickLabel = RF.PredictorNames; h.XTick = 1:30; h.XTickLabelRotation = 45; h.TickLabelInterpreter = 'none';
%}
 
%% Compare performance of LR and RF algorithms
% Plot ROC chart and compare AUC, PPV and NPV for LR and RF
figure; hold on
% Logistic regression:
scores = bestLR.results{bestLR.idxBest(1),1}.scores;
[xLR,yLR,~,AUC_LR] = perfcurve(y_test,scores(:,2),1);
plot(xLR,yLR,'color','r')
% Random Forest:
scores = bestRF.results{bestRF.idxBest(1),1}.scores;
[xRF,yRF,~,AUC_RF] = perfcurve(y_test,scores(:,2),1);
plot(xRF,yRF,'color','b')
legend(['LR (AUC=', num2str(AUC_LR), ')'],['RF (AUC=', num2str(AUC_RF), ')'])
title('ROC chart and Area Under Curve (AUC)')
xlabel('False positive rate'); ylabel('True positive rate');
 
% Future work:
% - Calibrate probabilities to improve model performance / identify optimal threshold
% - Adjust cost matrix to bias sensitivity vs. selectivity
% - Perform significance test for LR vs. RF model performance
% - Compare model results against Bayesian error limit
