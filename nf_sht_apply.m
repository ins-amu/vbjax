function x1 = nf_sht_apply(Lp, x0)
% Computes SHT-based local coupling on x0 using precomputed
% matrices Lp.

% with pagemtimes, timing similar to jax/numpy einsum

nlon = size(x0, 1);
nlat = size(x0, 2);
lmax = size(Lp,3);

% longitudinal Fourier transform
X = fft(x0, nlon, 1)';

% apply precomputed matrices for the forward (latitudinal) Legendre
% transform + diffusion + inverse Legendre transform
Xp = reshape(X, [nlat 1 nlon]);
LX = pagemtimes(Lp, Xp(:,:,1:lmax));
X(:, 1:lmax) = reshape(LX, [nlat lmax]);

% only applies up to m<=lmax, truncate rest by zeroing
X(:, lmax+1:end) = 0;

% back to real space
x1 = real(ifft(X', nlon, 1));