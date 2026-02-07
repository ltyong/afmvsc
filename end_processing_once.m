function [res, onetime] = end_processing_once(C,Y)
C = C ./ repmat(sqrt(sum(C.^2, 2)), 1,size(C,2));
tic;
pred = litekmeans(C,length(unique(Y)),'MaxIter',100, 'Replicates',1);    
%pred = kmeans(C,length(unique(Y)),'maxiter',1000,'replicates',50,'EmptyAction','singleton'); 
pred = pred(:);
onetime = toc;
resall = Clustering8Measure(Y,pred);
res = resall([1,2,3,4,7]);
end