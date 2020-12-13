clear all; close all; clc;
rng('default') 
 
data = readmatrix("IRIS.csv",'Range','A:E');
class = readmatrix("IRIS.csv",'Range','E:E');

percentTrainingData = 0.8;
[trainingData, testData, trainingClass, testClass] = splitData(data,percentTrainingData);

numTests = length(testData);
numClasses = max(trainingClass);
numVariables = size(trainingData,2);

%% Gaussian classifier
mu = zeros(numClasses,numVariables);
sigma = mu;
likelihoodTimesPrior = zeros(numTests,numClasses);
probability = likelihoodTimesPrior;
prior = zeros(1,numClasses);
marginal = zeros(1,numTests);

% Caculate mu and sigma from data set
for i=1:numClasses
    [mu(i,:) sigma(i,:)] = findInfoForClass(trainingData,trainingClass,i);
    prior(i) = findPrior(trainingData,trainingClass,i);
end

% Classify!
for i=1:numTests
    for j=1:numClasses
        likelihoodTimesPrior(i,j) = mvnpdf(testData(i,:),mu(j,:),sigma(j,:)).*prior(j);
    end
    marginal(i) = findMarginal(prior,likelihoodTimesPrior(i,:));
    probability(i,:) = likelihoodTimesPrior(i,:)./marginal(i);
end
 
%% Plot!
close all
color = ['r';'g';'b'];
s=zeros(1,numClasses);
for i=1:numClasses
    classData = data(data(:,end)==i,:);
    sepalLength = classData(:,1);
    sepalWidth = classData(:,2);
    petalLength = classData(:,3);
    petalWidth = classData(:,4);
    
    sx = petalLength;
    sy = petalWidth;
    
    hold on 
    [x, y] = meshgrid(sort(sx),sort(sy));
    z = mvnpdf([x(:) y(:)],mu(i,3:4),sigma(i,3:4));
    z = reshape(z,size(x));
    gmPDF = @(x,y) arrayfun(@(x0,y0) mvnpdf([x(:) y(:)],mu(i,3:4),sigma(i,3:4)),x,y);
    fcontour(gmPDF)
    s(i) = scatter(petalLength,petalWidth,'filled',color(i),'d');
    colorbar
    grid on
    hold off
end

grid on
xlim([1, 7])
ylim([0, 2.7])
title('Petal data');
xlabel('Length (cm)');
ylabel('Width (cm)');
legend([s(1) s(2) s(3)],{'Setosa','Versicolor','Virginica'},'Location','southeast')
%% Functions
 
% This function splits the data into a training set and testing set.
% @param data: the entire data set
% @param p: the portion that will be used as trainingData (recommended
% 0.5-0.8 depending on data set size)
% @return: vectors containing the training and test data set and the class
% labesl for the training and test data sets
function [trainingData, testData, trainingClass, testClass] = splitData(data,p)
r = rand(length(data),1);
trainingData = data(r<p,[1:end-1]);
testData = data(r>=p,[1:end-1]);
trainingClass = data(r<p,end);
testClass = data(r>=p,end);
end
 
% This function finds the prior probability.
% @param trainingData: the training data set
% @param data: the entire data set
% @return: the prior probability
function prior = findPrior(trainingData,trainingClass,classNum)
classData = trainingData(trainingClass==classNum,:); 
prior = length(classData)/length(trainingData);
end
 
% This function finds the marginal likelihood.
function marginal = findMarginal(prior,likelihoodTimesPrior)
    marginal = 0;
    for i=1:size(prior,2)
       marginal = marginal + likelihoodTimesPrior(i); 
    end
    
end
 
% This function finds the mu and sigma for chosen class
% @param data: the entire data set
% @param classNum: the class to find mu and sigma for
% @return: the mu and sigma for chosen class
function [mu sigma] = findInfoForClass(trainingData,trainingClass,classNum)
classData = trainingData(trainingClass==classNum,:); 
for j=1:size(classData,2)
   mu(j) = mean(classData(:,j)); 
   sigma(j) = var(classData(:,j)); 
end
end
 

 
