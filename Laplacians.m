function L = Laplacians(X, knn_para)
W = construct_adjacency_matrix(X, knn_para);
L = construct_laplacian_matrix(W);
end

function W = construct_adjacency_matrix(X, k)  % KNN
n = size(X, 1);
W = zeros(n, n);
for i = 1:n
    distances = sum((X - X(i, :)).^2, 2);
    [~, idx] = sort(distances);
    W(i, idx(2:k+1)) = 1; 
    W(idx(2:k+1), i) = 1; 
end
end

function L = construct_laplacian_matrix(W)
D = diag(1./sqrt(sum(W)+eps));
L = eye(length(W))-D*W*D;
end