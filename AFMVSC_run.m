function [result,S,time] = AFMVSC_run(X,L,lambda1,lambda2,lambda3,d,numsample)
%% initialize
maxIter = 20;
numview = length(X);

P = cell(numview,1);
A = zeros(d,d);
C = zeros(d,numsample);
G = zeros(d,numsample);
for i = 1:numview      
   di = size(X{i},1); 
   P{i} = zeros(di,d);
end

alpha = ones(1,numview)/numview;
% in advance
L_sum = zeros(size(L{1}));
for j=1:numview   
    L_sum = L_sum + L{j};
end
tempL = L_sum+lambda2*eye(length(L_sum));
Q = inv(tempL);
q = lambda3*ones(numsample,1)-2*tempL*ones(numsample,1); % only calculate once 
q = q/d;

flag = 1;
iter = 0;
tic;
while flag
    iter = iter + 1;
    
    %% optimize P_i
    AC = A*C;
    parfor iv=1:numview        
        tempP = X{iv}*AC';              
        [U,~,V] = svd(tempP,'econ');
        P{iv} = U*V';
    end

    %% optimize A
    sumAlpha = 0;
    part1 = 0;
    for ia = 1:numview
        al2 = alpha(ia)^2;
        sumAlpha = sumAlpha + al2;
        part1 = part1 + al2 * P{ia}' * X{ia};
    end
    part1 = part1 * C';
    [Unew,~,Vnew] = svd(part1,'econ');
    A = Unew*Vnew';
  
    %% optimize C
    % QP
    H = 2*sumAlpha*eye(d)+2*lambda1*eye(d);
    H = (H+H')/2;
    options = optimset('Algorithm','interior-point-convex','Display','off'); % Algorithm 默认为 interior-point-convex   
    ff_all=0;
    for j=1:numview
        ff_all = ff_all - 2*alpha(j)^2*A'*P{j}'*X{j};
    end
    ff_all = ff_all - lambda3*G;
    parfor ji=1:numsample
        ff=ff_all(:,ji);
        C(:,ji) = quadprog(H,ff,[],[],ones(1,d),1,zeros(d,1),ones(d,1),[],options);
    end

    %% optimize G  
    G = max(0,0.5*(lambda3*C-ones(d,1)*q')*Q);    
    G = G./sum(G,1);
    
    %% optimize alpha
    M = zeros(numview,1);
    parfor iv = 1:numview
        M(iv) = norm( X{iv} - P{iv} * A * C,'fro')^2;
    end
    Mfra = M.^-1;
    T = 1/sum(Mfra);
    alpha = T*Mfra;
    %%
    %
    term1 = 0;
    term3 = 0;
    for iv = 1:numview
        term1 = term1 + alpha(iv)^2 * norm(X{iv} - P{iv} * A * C,'fro')^2;        
        term3 = term3 + trace(G * L{iv} * G');  
    end
    term2 = lambda1 * norm(C,'fro')^2;
    term4 = lambda2 * norm(G,'fro')^2;
    term5 = lambda3 * trace(G * C');
    obj(iter) = term1+term2+term3+term4-term5;       
    %fprintf('Iter: %d, loss: %.11f.\n',iter,obj(iter));
    %}
    %if (iter>9) && (abs(obj(iter-1)-obj(iter))<abs(obj(iter-1))*1e-3 || iter>maxIter || abs(obj(iter)) < 1e-10)       
    if (iter>9)    
        flag = 0;
    end
end
S = (C+G)./2;
[~,~,VS]=svd(S,'econ');
result = VS;
time = toc;
end
         
         
    
