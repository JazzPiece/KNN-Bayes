clear all;
close all;
clc;rng('default') ;


data = readmatrix("IRIS_Num.csv",'Range','A:E');
class = readmatrix("IRIS_Num.csv",'Range','E:E');

percentTrainingData = 0.6;
[trainingData, testData, trainingClass, testClass] = splitData(data,percentTrainingData);

numTests = length(testData);
numClass = max(trainingClass);
numTrain = length(trainingData);
numVariables = size(trainingData,2);


%figure
%gscatter(data(:,1),data(:,3),class)

%figure
%gscatter(trainingData(:,1),trainingData(:,3),trainingClass)

%figure
%gscatter(trainingData(:,1),trainingData(:,2),trainingClass)
%% KNN classifier

rate = zeros(3,1);

for K = 1: 33
accuracy = zeros(numTests,3);

for j = 1:numTests
    distanceList = zeros(size(trainingData(1,:)));
    input = testData(j,:);
    for i = 1: numTrain
        distance = sqrt(sum((trainingData(i,:) - input) .^ 2));
        distanceList(i,1) = distance;
        distanceList(i,2) = trainingClass(i,1);
    end
    [B,I] = sortrows(distanceList(:,1));
    
    result = zeros(K);
    for c = 1: K
        result(c,1) = trainingClass(I(c,1),1);
    end
 
    hold = tabulate(result(1:K,1));
    rowSize =  length(hold(:,1));
    [B,I] = sortrows(hold(:,3));
    index = I(rowSize);
   
    accuracy(j,1) = hold(rowSize,1);
    accuracy(j,2) = hold(index,3);
    
    if  testClass(j) == accuracy(j,1)
        accuracy(j,3) = 1;
    end
    
end
tabulate(accuracy(1:numTests,3))
hold2 = tabulate(accuracy(1:numTests,3));
rate(K,1) = hold2(2,3);
end 

Accuracy = rate 

figure
gscatter(1:33,Accuracy)

%figure
%gscatter(1:numTests,accuracy(:,2),accuracy(:,3))



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

 
