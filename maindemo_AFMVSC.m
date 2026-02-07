clear;
clc;
warning off;
addpath(genpath('./'));
ds{1} = 'Mfeat';  % [3c 18 7 25]
dsPath = './datasets/';
dataName = ds{1}; 
disp(dataName);
load(strcat(dsPath,dataName));
X=data; Y=truelabel{1};  % Mfeat, 100leaves, Hdigit
%X=cell(2,1); X{1}=sport01; X{2}=sport02; Y=truth;  % BBCSport
numsample = length(Y);
c = length(unique(Y));
L = cell(length(X),1);
for i = 1:length(X) 
   X{i} = double(X{i});  
   %X{i} = X{i}';   % ensure size m_i*n
   %X{i} = mapstd(X{i},0,1);  % 100leaves, ALOI-100
   X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1); 
   %X{i} = X{i}./(repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1)+10e-10); % Reuters, AWA
   L{i} = Laplacians(X{i}',5);   
end
lambda1 = 18; 
lambda2 = 7;
lambda3 = 25; 
k = 3*c;  % Testing running time should use c
for j=1:1
[V,S,time1]=AFMVSC_run(X,L,lambda1,lambda2,lambda3,k,numsample);
%[~, time2] = end_processing_once(V,Y);  % Test running time of one k-means
[res, std, time2] = end_processing_mean(V,Y);
fprintf('Clusters:%d, Time:%.6f\n',[c, time1+time2]);
fprintf('ACC:%.6f, NMI:%.6f, Purity:%.6f, Fscore:%.6f, AR:%.6f\n',res);
fprintf('ACC_std:%.6f, NMI_std:%.6f, Purity_std:%.6f, Fscore_std:%.6f, AR_std:%.6f\n',std);
end