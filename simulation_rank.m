function results = simulation_rank(r)

addpath(genpath('tensor_toolbox_2.6'));

lambda = 0;
run_time = 100;
rx = 1; ry = r;
results = zeros(3, 10);

time1 = zeros(3, run_time);
Similarity_U1 = zeros(3, run_time);
Similarity_U2 = zeros(3, run_time);
Similarity_V1 = zeros(3, run_time);
Similarity_V2 = zeros(3, run_time);

for run = 1:run_time
    
    %% Synthetic Data
    
    % 1D X true signal: 100-by-1 vector
    V = zeros(100, 1);
    V([13 24 57 73 92]) = 1;
    
    W = zeros(64, 64);
    W(15:49, 28:36) = 1;
    W(28:36, 15:49) = 1;
    [p1, p2] = size(W);
    
    % Group 1
    V1 = V;
    V1([8 66]) = 0.8;
    W1 = W;
    W1(13:23, 13:23) = 0.9;
    W1(45:55, 45:55) = 0.7;
    
    % Group 2
    V2 = V;
    V2([47 86]) = 0.6;
    W2 = W;
    W2(55:60, 5:20) = 0.8;
    W2(5:10, 40:55) = 0.8;
    
    % simulate joint normal random deviates
    n1 = 1000; n2 = 1000;
    rho1 = 0.95; rho2 = 0.97;
    
    s = RandStream('mt19937ar','Seed',1);
    RandStream.setGlobalStream(s);
    
    [X1, Y1] = simcca(V1, W1(:), rho1, n1, 'noisex', 1, 'noisey', 1e-3);
    [X2, Y2] = simcca(V2, W2(:), rho2, n2, 'noisex', 1, 'noisey', 1e-3);

    [X1t, Y1t] = simcca(V1, W1(:), rho1, 200, 'noisex', 1, 'noisey', 1e-3);
    [X2t, Y2t] = simcca(V2, W2(:), rho2, 200, 'noisex', 1, 'noisey', 1e-3);
    
    X = [X1; X2];
    Y = [Y1; Y2];
    
    % make data into tensors
    if ~isa(X, 'tensor')
      X = tensor(X');
      Y = tensor(Y', [p1 p2 n1+n2]);
    end
    
    if ~isa(X1, 'tensor')
      X1 = tensor(X1');
      X1t = tensor(X1t');
      Y1 = tensor(Y1', [p1 p2 n1]);
      Y1t = tensor(Y1t', [p1 p2 200]);
    end
    
    if ~isa(X2, 'tensor')
      X2 = tensor(X2');
      X2t = tensor(X2t');
      Y2 = tensor(Y2', [p1 p2 n2]);
      Y2t = tensor(Y2t', [p1 p2 200]);
    end

    %% Overall STCCA

%     t1 = tic;
%     [betax, betay, ~] = tcca(X, Y, rx, ry, lambda, ...
%         'xl0maxprop', 0.8, 'yl0maxprop', 0.8);
%     
%     STCCA_U1 = (abs(sum((V1(:) / norm(V1(:))) .* (double(betax(:)) / norm(double(betax(:)))))));
%     STCCA_V1 = (abs(sum((W1(:) / norm(W1(:))) .* (vec(double(full(betay))) / norm(betay)))));
% 
%     STCCA_U2 = (abs(sum((V2(:) / norm(V2(:))) .* (double(betax(:)) / norm(double(betax(:)))))));
%     STCCA_V2 = (abs(sum((W2(:) / norm(W2(:))) .* (vec(double(full(betay))) / norm(betay)))));
% 
%     time1(1, run) = toc(t1);
%     clear t1;
%     Similarity_U1(1, run) = STCCA_U1;
%     Similarity_V1(1, run) = STCCA_V1;
%     Similarity_U2(1, run) = STCCA_U2;
%     Similarity_V2(1, run) = STCCA_V2;


    %% MG-STCCA
    X_lt = cell(2, 1); Y_lt = cell(2, 1);
    X_lt{1} = X1; X_lt{2} = X2;
    Y_lt{1} = Y1; Y_lt{2} = Y2;
    
    t1 = tic;
    [betax, betay, ~] = gtcca(X_lt, Y_lt, rx, ry, lambda, ...
        'xl0maxprop', 0.6, 'yl0maxprop', 0.6, 'alpha', 0.4, 'sigma', 0.4);
    
%     betax1 = betax{1}; betax2 = betax{2};
%     betay1 = betay{1}; betay2 = betay{2};
%     
%     GTCCA_U1 = (abs(sum((V1(:) / norm(V1(:))) .* (double(betax1(:)) / norm(double(betax1(:)))))));
%     GTCCA_V1 = (abs(sum((W1(:) / norm(W1(:))) .* (vec(double(full(betay1))) / norm(betay1)))));
%     
%     GTCCA_U2 = (abs(sum((V2(:) / norm(V2(:))) .* (double(betax2(:)) / norm(double(betax2(:)))))));
%     GTCCA_V2 = (abs(sum((W2(:) / norm(W2(:))) .* (vec(double(full(betay2))) / norm(betay2)))));

    time1(1, run) = toc(t1);
    clear t1;

%     Similarity_U1(2, run) = GTCCA_U1;
%     Similarity_V1(2, run) = GTCCA_V1;
%     Similarity_U2(2, run) = GTCCA_U2;
%     Similarity_V2(2, run) = GTCCA_V2;
    
    %% Group 1 STCCA
    
    t1 = tic;
    [betax3, betay3, ~] = tcca(X1, Y1, rx, ry, lambda, ...
        'xl0maxprop', 0.8, 'yl0maxprop', 0.8);
    time1(2, run) = time1(2, run) + toc(t1);
    clear t1;
    
%     Ind_U1 = (abs(sum((V1(:) / norm(V1(:))) .* (double(betax3(:)) / norm(double(betax3(:)))))));
%     Ind_V1 = (abs(sum((W1(:) / norm(W1(:))) .* (vec(double(full(betay3))) / norm(betay3)))));
    
    %% Group 2 STCCA
    t1 = tic;
    [betax4, betay4, ~] = tcca(X2, Y2, rx, ry, lambda, ...
        'xl0maxprop', 0.8, 'yl0maxprop', 0.8);
    time1(2, run) = time1(2, run) + toc(t1);
    clear t1;
    
%     Ind_U2 = (abs(sum((V2(:) / norm(V2(:))) .* (double(betax4(:)) / norm(double(betax4(:)))))));
%     Ind_V2 = (abs(sum((W2(:) / norm(W2(:))) .* (vec(double(full(betay4))) / norm(betay4)))));
% 
%     Similarity_U1(3, run) = Ind_U1;
%     Similarity_V1(3, run) = Ind_V1;
%     Similarity_U2(3, run) = Ind_U2;
%     Similarity_V2(3, run) = Ind_V2;

    %% Group 1 TCCA
    
    t1 = tic;
    [betax3, betay3, ~] = tcca(X1, Y1, rx, ry, lambda);
    time1(3, run) = time1(3, run) + toc(t1);
    clear t1;
    
     %% Group 2 TCCA
    t1 = tic;
    [betax4, betay4, ~] = tcca(X2, Y2, rx, ry, lambda);
    time1(3, run) = time1(3, run) + toc(t1);
    clear t1;

end

results(1,1) = mean(time1(1,:));
results(1,2) = std(time1(1,:));
% results(1,3) = mean(Similarity_U1(1,:));
% results(1,4) = std(Similarity_U1(1,:));
% results(1,5) = mean(Similarity_V1(1,:));
% results(1,6) = std(Similarity_V1(1,:));
% results(1,7) = mean(Similarity_U2(1,:));
% results(1,8) = std(Similarity_U2(1,:));
% results(1,9) = mean(Similarity_V2(1,:));
% results(1,10) = std(Similarity_V2(1,:));

results(2,1) = mean(time1(2,:));
results(2,2) = std(time1(2,:));
% results(2,3) = mean(Similarity_U1(2,:));
% results(2,4) = std(Similarity_U1(2,:));
% results(2,5) = mean(Similarity_V1(2,:));
% results(2,6) = std(Similarity_V1(2,:));
% results(2,7) = mean(Similarity_U2(2,:));
% results(2,8) = std(Similarity_U2(2,:));
% results(2,9) = mean(Similarity_V2(2,:));
% results(2,10) = std(Similarity_V2(2,:));
% 
% results(3,3) = mean(Similarity_U1(3,:));
% results(3,4) = std(Similarity_U1(3,:));
% results(3,5) = mean(Similarity_V1(3,:));
% results(3,6) = std(Similarity_V1(3,:));
% results(3,7) = mean(Similarity_U2(3,:));
% results(3,8) = std(Similarity_U2(3,:));
% results(3,9) = mean(Similarity_V2(3,:));
% results(3,10) = std(Similarity_V2(3,:));

results(3,1) = mean(time1(3,:));
results(3,2) = std(time1(3,:));
writematrix(results, [num2str(r) 'results.csv'])

end