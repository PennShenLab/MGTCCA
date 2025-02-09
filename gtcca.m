function [betax, betay, rho] = gtcca(X, Y, rx, ry, lambda, varargin)
% 
% Group Sparse Tensor Canonical Correlation Analysis
% 
% Input:
%     X: list of p1-by-p2-by-...-by-px-by-n tensor data
%     Y: list of q1-by-q2-by-...-by-qy-by-n tensor data
%     rx: rank of X signal
%     ry: rank of Y signal
%     alpha, sigma: balance between the element-wise and row-wise sparsity levels
%     xl0maxprop: proportion of nonzero element selected for betax
%     yl0maxprop: proportion of nonzero element selected for betay
%     
%Optional:
%     betax0: list of p1-by-p2-by-...-by-px tensors as initial canonical tensors for tensor data X
%     betay0: list of q1-by-q2-by-...-by-qy tensors as initial canonical tensors for tensor data Y
%     
%Output:
%     betax: list of rank rx (p1,...,px) tensor signal
%     betay: list of rank ry (p1,...,py) tensor signal
%     rho: canonical correlation


%% parse inputs
argin = inputParser;
argin.addRequired('X');
argin.addRequired('Y');

argin.addRequired('rx', @(x) x>=1);
argin.addRequired('ry', @(x) x>=1);
argin.addRequired('lambda', @(x) x>=0);

argin.addParameter('betax0', [], @isnumeric);
argin.addParameter('betay0', [], @isnumeric);
argin.addParameter('alpha', 0.2, @(x) isnumeric(x) && x>0);
argin.addParameter('sigma', 0.2, @(x) isnumeric(x) && x>0);

argin.addParameter('tolfun', 1e-4, @(x) isnumeric(x) && x>0);
argin.addParameter('maxiter', 100, @(x) isnumeric(x) && x>0);
argin.addParameter('replicates', 5, @(x) isnumeric(x) && x>0);

argin.addParameter('xl0maxprop', 0.8, @(x) isnumeric(x) && x>0);
argin.addParameter('yl0maxprop', 0.8, @(x) isnumeric(x) && x>0);
argin.parse(X, Y, rx, ry, lambda, varargin{:});

alpha = argin.Results.alpha;
sigma = argin.Results.sigma;

betax0 = argin.Results.betax0;
betay0 = argin.Results.betay0;

tolfun = argin.Results.tolfun;
maxiter = argin.Results.maxiter;

xl0maxprop = argin.Results.xl0maxprop;
yl0maxprop = argin.Results.yl0maxprop;

if isempty(betax0)
  replicates = argin.Results.replicates;
else
  replicates = 1;
end

%% prepare
X_raw = X;
Y_raw = Y;

% retrieve dimensions
[dx, dy, px, py, ~] = retrieve_dimensions(X_raw{1}, Y_raw{1});

for g = 1:size(X_raw, 1)
    X = X_raw{g};
    Y = Y_raw{g};

    % convert data (X, Y) into tensors
    X = tensor(X);
    Y = tensor(Y);

    % center data
    Xc = double(tenmat(X, dx+1, 't'));
    Yc = double(tenmat(Y, dy+1, 't'));
    Yc = tensor(bsxfun(@minus, Yc, mean(Yc, 2)), size(Y));
    Xc = tensor(bsxfun(@minus, Xc, mean(Xc, 2)), size(X));
    X_raw{g} = Xc;
    Y_raw{g} = Yc;

    clear X Y;
end

% pre-compute sample covariances
varxy = cell(size(X_raw, 1), 1);
varxx = cell(size(X_raw, 1), 1);
varyy = cell(size(X_raw, 1), 1);

for g = 1:size(X_raw, 1)
    Xc = X_raw{g};
    Yc = Y_raw{g};

    varxy{g} = cell(dx, dy);
    varxx{g} = cell(dx, 1);
    varyy{g} = cell(dy, 1);

    [~, ~, ~, ~, n] = retrieve_dimensions(Xc, Yc);

    for i = 1:dx
        Xi = double(tenmat(Xc, [i 1:i-1 i+1:dx], dx+1));

        % estimate Cov(Xi)
        varxx{g}{i} = (Xi * Xi') / n;

        for j = 1:dy
            Yj = double(tenmat(Yc, [j 1:j-1 j+1:dy], dy+1));
            if i == 1
                % estimate Cov(Yj)
                varyy{g}{j} = (Yj * Yj') / n;
            end
            % estimate Cov(Xi, Yj)
            varxy{g}{i,j} = (Xi * Yj') / n;
        end
    end
end
clear Xi Yj;

%% main loop
rho_best = cell(size(X_raw, 1), 1);
betax_best = cell(size(X_raw, 1), 1);
betay_best = cell(size(X_raw, 1), 1);

for g = 1:size(X_raw, 1)
    rho_best{g} = -inf;
end

for rep = 1:replicates

    % starting point
    if isempty(betax0)
        betax = cell(size(X_raw, 1), 1);
        for g = 1:size(X_raw, 1)
            betax{g} = ktensor(arrayfun(@(j) rand(px(j),rx), 1:dx, 'UniformOutput',false));
        end
    else
        betax = betax0;
    end

    if isempty(betay0)
        betay = cell(size(X_raw, 1), 1);
        for g = 1:size(X_raw, 1)
            betay{g} = ktensor(arrayfun(@(j) rand(py(j),ry), 1:dy, 'UniformOutput',false));
        end
    else
        betay = betay0;
    end

    objval = cell(size(X_raw, 1), 1);
    objval_old = cell(size(X_raw, 1), 1);
    for g = 1:size(X_raw, 1)
        objval{g} = -inf;
        objval_old{g} = 0;
    end

    for iter = 1:maxiter
        % mode of X
        i = mod(iter-1, dx) + 1;
        % mode of Y
        j = mod(iter-1, dy) + 1;

        % initialize cumulative Katri-Rao product
        if i == 1
            cumkrx = cell(size(X_raw, 1), 1);
            for g = 1:size(X_raw, 1)
                cumkrx{g} = ones(1, rx);
            end
        end
        if j == 1
            cumkry = cell(size(X_raw, 1), 1);
            for g = 1:size(X_raw, 1)
                cumkry{g} = ones(1, ry);
            end
        end

        for g = 1:size(X_raw, 1)

            % compute the covariance matrices in subproblem
            if i == dx
                matlt = kron(cumkrx{g}', eye(px(i)));
            else
                matlt = kron(khatrirao([betax{g}.U(dx:-1:i+1),cumkrx{g}])', eye(px(i)));
            end
            if j == dy
                matrt = kron(cumkry{g}, eye(py(j)));
            else
                matrt = kron(khatrirao([betay{g}.U(dy:-1:j+1),cumkry{g}]), eye(py(j)));
            end
            Cxy = matlt * varxy{g}{i,j} * matrt;
            Cxx = matlt * varxx{g}{i} * matlt';
            Cyy = matrt' * varyy{g}{j} * matrt;

            % add ridge to Cxx and Cyy
            Cxx = Cxx + 1e-6 * speye(size(Cxx, 1));
            Cyy = Cyy + 1e-6 * speye(size(Cyy, 1));

            if sum(sum(isnan(Cyy))) > 0
                break;
            end

            % solve the subproblem by generalized eigenvalue problem
            A = sparse([zeros(size(Cxx)) Cxy; Cxy' zeros(size(Cyy))]);
            B = sparse([Cxx zeros(size(Cxy)); zeros(fliplr(size(Cxy))), Cyy]);

            objval_old{g} = objval{g};
            [evec, objval{g}] = eigs(A, B, 1, 'lm');

            betax{g}{i} = sign(objval{g}) * reshape(evec(1:(px(i)*rx)), px(i), rx);
            betay{g}{j} = reshape(evec((px(i)*rx+1):end), py(j), ry);
            objval{g} = abs(objval{g});
        end

        % enforce cardinality constraint
        l2xrownorm = zeros(numel(betax{1}{i}), 1);
        l2yrownorm = zeros(numel(betay{1}{j}), 1);
        for g = 1:size(X_raw, 1)
            l2xrownorm = l2xrownorm + betax{g}{i}(:).^2;
            l2yrownorm = l2yrownorm + betay{g}{j}(:).^2;
        end
        l2xrownorm = l2xrownorm.^(1/2);
        l2yrownorm = l2yrownorm.^(1/2);

        for g = 1:size(X_raw, 1)
            xisort = sort((1-alpha)*abs(betax{g}{i}(:))+alpha*l2xrownorm, 'descend');
            yjsort = sort((1-sigma)*abs(betay{g}{j}(:))+sigma*l2yrownorm, 'descend');
            xibenm = xisort(min(floor(xl0maxprop*px(i)*rx), px(i)*rx));
            yjbenm = yjsort(min(floor(yl0maxprop*py(j)*ry), py(j)*ry));
            betax{g}{i}(abs(betax{g}{i}) < xibenm) = 0;
            betay{g}{j}(abs(betay{g}{j}) < yjbenm) = 0;
        end

        % accumulate Katri-Rao products
        for g = 1:size(X_raw, 1)
            cumkrx{g} = khatrirao(betax{g}{i}, cumkrx{g});
            cumkry{g} = khatrirao(betay{g}{j}, cumkry{g});
        end

        % check stopping criterion
        err = abs(mean(cell2mat(objval)) - mean(abs(cell2mat(objval_old))));
        if err < tolfun * (mean(abs(cell2mat(objval_old))) + 1) && iter > max(dx, dy)
            break;
        end
    end
       
    % record if we have a better correlation
    for g = 1:size(X_raw, 1)
        Xc = X_raw{g};
        Yc = Y_raw{g};
        canvarx = double(ttt(Xc, tensor(betax{g}), 1:dx));
        canvary = double(ttt(Yc, tensor(betay{g}), 1:dy));
        rho_rep = corr(canvarx, canvary);

        if rho_rep > rho_best{g}
            rho_best{g} = rho_rep;
            betax_best{g} = betax{g};
            betay_best{g} = betay{g};
        end
    end
end

rho = rho_best;
betax = betax_best;
betay = betay_best;