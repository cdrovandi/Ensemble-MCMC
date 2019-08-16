function [X1, X2, X3] = simulate_Lorenz63_single_frn(x, theta, noise_scale, dt, steps, q)
%%
% FIXED RANDOM NUMBERS
% inputs:
%   x - current state vector (vector of length 3)
%   theta - parameters (vector of length 3)
%   noise_scale - scale of noise to add (scalar)
%   dt - time gap
%
% outputs:
%   X1, X2, X3 - updated state vector
%%

X1 = x(1); X2 = x(2); X3 = x(3);

theta1 = theta(1); theta2 = theta(2); theta3 = theta(3);

for t = 1:steps
  drift1 = theta1 * (X2 - X1);
  drift2 = theta2 * X1 - X2 - X1 * X3;
  drift3 = X1 * X2 - theta3 * X3;
  X1 = X1 + drift1 * dt + sqrt(dt) * noise_scale * q(t,1);
  X2 = X2 + drift2 * dt + sqrt(dt) * noise_scale * q(t,2);
  X3 = X3 + drift3 * dt + sqrt(dt) * noise_scale * q(t,3);
end
end
