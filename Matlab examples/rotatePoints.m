function [xs_rot, ys_rot] = rotatePoints(x, y, x0, y0, theta)
% Rotate the points an angle of theta along the central point (x0, y0)
    x = x - x0;
    y = y - y0;
    xs_rot = x .* cos(theta) - y .* sin(theta) + x0;
    ys_rot = x .* sin(theta) + y .*cos(theta) + y0;
end