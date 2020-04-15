function [x] = simulate_Lorenz96_single(x, n, theta, noise_scale, dt, steps)
%%
% inputs:
%   x - current state vector
%   n - state dimension (at least 4)
%   theta - parameters (vector of length 3)
%   noise_scale - scale of noise to add (scalar)
%   dt - time gap
%
% outputs:
%   x - updated state vector
%%

for t = 1:steps
  drift(1) = theta(1) * (x(2) - x(n-1))*x(n) - theta(2) * x(1) + theta(3);
  drift(2) = theta(1) * (x(3) - x(n))*x(1) - theta(2) * x(2) + theta(3);
  for i = 3:(n-1)
    drift(i) = theta(1) * (x(i+1) - x(i-2))*x(i-1) - theta(2) * x(i) + theta(3);
  end
  drift(n) = theta(1) * (x(1) - x(n-2))*x(n-1) - theta(2) * x(n) + theta(3);
  x = x + drift * dt + sqrt(dt) * noise_scale * randn(1,n);
end
end
