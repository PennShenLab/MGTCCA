function [dx, dy, px, py, n] = retrieve_dimensions(X, Y)

dx = ndims(X) - 1;
dy = ndims(Y) - 1;
px = size(X);
py = size(Y);
n = px(end);

end