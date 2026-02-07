function [resmean, resstd, onetime] = end_processing_mean(C,Y)
maxIter = 20;
C = C ./ repmat(sqrt(sum(C.^2, 2)), 1, size(C,2));
times = zeros(maxIter,1);
pred = cell(maxIter,1);
for iter = 1:maxIter
    tic;    
    % Test running times uses 1
    %temp = litekmeans(C,length(unique(Y)),'MaxIter',100, 'Replicates',1); % Caltech101-all, AWA   
    temp = kmeans(C,length(unique(Y)),'maxiter',1000,'replicates',50,'EmptyAction','singleton');
    pred{iter} = temp;
    times(iter) = toc;
    measurements(iter,:) = Clustering8Measure(Y,pred{iter});
    %fprintf('ACC:%.6f, NMI:%.6f, Purity:%.6f, Fscore:%.6f, Precision:%.6f, Recall:%.6f, AR:%.6f\n',measurements(iter,1:7));
end
resmean = zeros(1,5);
resstd = zeros(1,5);
for i=1:4
    resmean(i) = mean(measurements(:,i));
    resstd(i) = std(measurements(:,i)); 
end
resmean(5) = mean(measurements(:,7));
resstd(5) = std(measurements(:,7));
onetime = mean(times);