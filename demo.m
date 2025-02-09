close all; clear; clc;
addpath(genpath('tensor_toolbox_2.6'));
    
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
% W1(28:36, 28:36) = 0;
W1(13:23, 13:23) = 0.9;
W1(45:55, 45:55) = 0.7;

% for i = 1:10
%     position1 = round(rand(1)*60)+1;
%     position2 = round(rand(1)*60)+1;
%     W1(position1:position1+3, position2:position2+3) = 0.9;
% end

% Group 2
V2 = V;
V2([47 86]) = 0.6;
W2 = W;
% W2(28:36, 28:36) = 0;
W2(55:60, 5:20) = 0.8;
W2(5:10, 40:55) = 0.8;

% for i = 1:10
%     position1 = round(rand(1)*60)+1;
%     position2 = round(rand(1)*60)+1;
%     W2(position1:position1+3, position2:position2+3) = 0.9;
% end


% 2D Y true signal: 64-by-64 cross
figure;
set(gca, 'FontSize', 15);
subplot(1, 2, 1)
hold on;
plot(double(V1), '+')
title('True V');
axis square;
subplot(1, 2, 2)
imagesc(W1);
colormap(gray);
title('True W');
axis square;

figure;
set(gca, 'FontSize', 15);
subplot(1, 2, 1)
hold on;
plot(double(V2), '+')
title('True V');
axis square;
subplot(1, 2, 2)
imagesc(W2);
colormap(gray);
title('True W');
axis square;


% simulate joint normal random deviates
n1 = 1000; n2 = 1000;
rho1 = 0.95; rho2 = 0.97;

s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

[X1, Y1] = simcca(V1, W1(:), rho1, n1, 'noisex', 1, 'noisey', 1e-3);
[X2, Y2] = simcca(V2, W2(:), rho2, n2, 'noisex', 1, 'noisey', 1e-3);

X = [X1; X2];
Y = [Y1; Y2];

% make data into tensors
if ~isa(X, 'tensor')
  X = tensor(X');
  Y = tensor(Y', [p1 p2 n1+n2]);
end

if ~isa(X1, 'tensor')
  X1 = tensor(X1');
  Y1 = tensor(Y1', [p1 p2 n1]);
end

if ~isa(X2, 'tensor')
  X2 = tensor(X2');
  Y2 = tensor(Y2', [p1 p2 n2]);
end

lambda = 0;
rx = 1; ry = 5;

%% Overall STCCA
[betax0, betay0, rho] = tcca(X, Y, rx, ry, lambda);

figure;
set(gca, 'FontSize', 15);
subplot(1, 2, 1)
hold on;
plot(abs(double(betax0)), 'o')
title({['Group1 STCCA, rank = (', num2str(rx) ', ' num2str(ry) ')']; ...
  ['\angle (V, Vhat) = ' num2str(abs(sum((V1(:) / norm(V1(:))) ...
  .* (double(betax0(:)) / norm(double(betax0(:)))))))]});
axis square;
subplot(1, 2, 2)
imagesc(double(full(betay0)));
colormap(gray);
title({['\rhohat = ' num2str(rho)]; ...
  ['\angle (W, What) = ' num2str(abs(sum((W1(:) / norm(W1(:))) ...
  .* (vec(double(full(betay0))) / norm(betay0)))))]});
axis square

STCCA_U1 = (abs(sum((V1(:) / norm(V1(:))) .* (double(betax0(:)) / norm(double(betax0(:)))))));
STCCA_V1 = (abs(sum((W1(:) / norm(W1(:))) .* (vec(double(full(betay0))) / norm(betay0)))));

figure;
set(gca, 'FontSize', 15);
subplot(1, 2, 1)
hold on;
plot(abs(double(betax0)), 'o')
title({['Group2 STCCA, rank = (', num2str(rx) ', ' num2str(ry) ')']; ...
  ['\angle (V, Vhat) = ' num2str(abs(sum((V2(:) / norm(V2(:))) ...
  .* (double(betax0(:)) / norm(double(betax0(:)))))))]});
axis square;
subplot(1, 2, 2)
imagesc(double(full(betay0)));
colormap(gray);
title({['\rhohat = ' num2str(rho)]; ...
  ['\angle (W, What) = ' num2str(abs(sum((W2(:) / norm(W2(:))) ...
  .* (vec(double(full(betay0))) / norm(betay0)))))]});
axis square

STCCA_U2 = (abs(sum((V2(:) / norm(V2(:))) .* (double(betax0(:)) / norm(double(betax0(:)))))));
STCCA_V2 = (abs(sum((W2(:) / norm(W2(:))) .* (vec(double(full(betay0))) / norm(betay0)))));

betay0 = double(full(betay0));
betax0 = double(full(betax0));

%% GS-TCCA
X_lt = cell(2, 1); Y_lt = cell(2, 1);
X_lt{1} = X1; X_lt{2} = X2;
Y_lt{1} = Y1; Y_lt{2} = Y2;

[betax, betay, rho] = gtcca(X_lt, Y_lt, rx, ry, lambda, ...
    'xl0maxprop', 0.4, 'yl0maxprop', 0.4, 'alpha', 0.3, 'sigma', 0.4);

betax1 = betax{1}; betax2 = betax{2};
betay1 = betay{1}; betay2 = betay{2};

figure;
set(gca, 'FontSize', 15);
subplot(1, 2, 1)
hold on;
plot(abs(double(betax1)), 'o')
title({['Group1 GS-TCCA, rank = (', num2str(rx) ', ' num2str(ry) ')']; ...
  ['\angle (V, Vhat) = ' num2str(abs(sum((V1(:) / norm(V1(:))) ...
  .* (double(betax1(:)) / norm(double(betax1(:)))))))]});
axis square;
subplot(1, 2, 2)
imagesc(double(full(betay1)));
colormap(gray);
title({['\rhohat = ' num2str(rho{1})]; ...
  ['\angle (W, What) = ' num2str(abs(sum((W1(:) / norm(W1(:))) ...
  .* (vec(double(full(betay1))) / norm(betay1)))))]});
axis square

GTCCA_U1 = (abs(sum((V1(:) / norm(V1(:))) .* (double(betax1(:)) / norm(double(betax1(:)))))));
GTCCA_V1 = (abs(sum((W1(:) / norm(W1(:))) .* (vec(double(full(betay1))) / norm(betay1)))));

figure;
set(gca, 'FontSize', 15);
subplot(1, 2, 1)
hold on;
plot(abs(double(betax2)), 'o')
title({['Group2 GS-TCCA, rank = (', num2str(rx) ', ' num2str(ry) ')']; ...
  ['\angle (V, Vhat) = ' num2str(abs(sum((V2(:) / norm(V2(:))) ...
  .* (double(betax2(:)) / norm(double(betax2(:)))))))]});
axis square;
subplot(1, 2, 2)
imagesc(double(full(betay2)));
colormap(gray);
title({['\rhohat = ' num2str(rho{2})]; ...
  ['\angle (W, What) = ' num2str(abs(sum((W2(:) / norm(W2(:))) ...
  .* (vec(double(full(betay2))) / norm(betay2)))))]});
axis square

GTCCA_U2 = (abs(sum((V2(:) / norm(V2(:))) .* (double(betax2(:)) / norm(double(betax2(:)))))));
GTCCA_V2 = (abs(sum((W2(:) / norm(W2(:))) .* (vec(double(full(betay2))) / norm(betay2)))));

betay1 = double(full(betay1));
betay2 = double(full(betay2));

betax1 = double(full(betax1));
betax2 = double(full(betax2));

%% Group 1 STCCA
[betax3, betay3, rho] = tcca(X1, Y1, rx, ry, lambda);

figure;
set(gca, 'FontSize', 15);
subplot(1, 2, 1)
hold on;
plot(abs(double(betax3)), 'o')
title({['Group1 STCCA, rank = (', num2str(rx) ', ' num2str(ry) ')']; ...
  ['\angle (V, Vhat) = ' num2str(abs(sum((V1(:) / norm(V1(:))) ...
  .* (double(betax3(:)) / norm(double(betax3(:)))))))]});
axis square;
subplot(1, 2, 2)
imagesc(double(full(betay3)));
colormap(gray);
title({['\rhohat = ' num2str(rho)]; ...
  ['\angle (W, What) = ' num2str(abs(sum((W1(:) / norm(W1(:))) ...
  .* (vec(double(full(betay3))) / norm(betay3)))))]});
axis square

Ind_U1 = (abs(sum((V1(:) / norm(V1(:))) .* (double(betax3(:)) / norm(double(betax3(:)))))));
Ind_V1 = (abs(sum((W1(:) / norm(W1(:))) .* (vec(double(full(betay3))) / norm(betay3)))));

%% Group 2 STCCA
[betax4, betay4, rho] = tcca(X2, Y2, rx, ry, lambda ...
    );

figure;
set(gca, 'FontSize', 15);
subplot(1, 2, 1)
hold on;
plot(abs(double(betax4)), 'o')
title({['Group2 STCCA, rank = (', num2str(rx) ', ' num2str(ry) ')']; ...
  ['\angle (V, Vhat) = ' num2str(abs(sum((V2(:) / norm(V2(:))) ...
  .* (double(betax4(:)) / norm(double(betax4(:)))))))]});
axis square;
subplot(1, 2, 2)
imagesc(double(full(betay4)));
colormap(gray);
title({['\rhohat = ' num2str(rho)]; ...
  ['\angle (W, What) = ' num2str(abs(sum((W2(:) / norm(W2(:))) ...
  .* (vec(double(full(betay4))) / norm(betay4)))))]});
axis square

Ind_U2 = (abs(sum((V2(:) / norm(V2(:))) .* (double(betax4(:)) / norm(double(betax4(:)))))));
Ind_V2 = (abs(sum((W2(:) / norm(W2(:))) .* (vec(double(full(betay4))) / norm(betay4)))));

betay3 = double(full(betay3));
betay4 = double(full(betay4));

betax3 = double(full(betax3));
betax4 = double(full(betax4));