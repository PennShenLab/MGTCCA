function [cor] = correlation(X, Y, betax, betay)

dx = ndims(X) - 1;
dy = ndims(Y) - 1;

canvarx = double(ttt(X, tensor(betax), 1:dx));
canvary = double(ttt(Y, tensor(betay), 1:dy));
cor = corr(canvarx, canvary);

end