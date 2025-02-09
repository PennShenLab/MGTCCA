function [Xc, Yc] = center_data(X, Y)

% convert data (X, Y) into tensors
X = tensor(X);
Y = tensor(Y);

% retrieve dimensions
dx = ndims(X) - 1;
dy = ndims(Y) - 1;

% center data
Xc = double(tenmat(X, dx+1, 't'));
Xc = tensor(bsxfun(@minus, Xc, mean(Xc, 2)), size(X));
Yc = double(tenmat(Y, dy+1, 't'));
Yc = tensor(bsxfun(@minus, Yc, mean(Yc, 2)), size(Y));

end