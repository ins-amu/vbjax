function [pmn,ml] = nf_sht_pmn(lmax, nlat) %, x)
% nf_sht_pmn generates Pmn coefficients for the Legendre
% transform step of the SHT.

% this function could do the Legendre Transform directly
% by using the commented lines w/ Q but it seems faster in
% MATLAB to use the matrix form with apply_L

% nlat = size(x,2);
% nlon = size(x,1);

%% setup ml
ml = zeros(2,lmax*(lmax-1));
ml_count = 0;
for m=0:lmax
    for l=m:lmax
        if m <= l
            ml_count = ml_count + 1;
            ml(1, ml_count) = m;
            ml(2, ml_count) = l;
        end
    end
end

%% setup amm
amm = zeros(1, lmax+1);
for m=0:lmax
    k = 1:m;
    if m > 0
        p = prod( (2*k+1) ./ (2*k) );
    else
        p = 1;
    end
    amm(m+1) = sqrt(p / (4*pi));
end
amm = single(amm);

%% setup cx
cx = single( cos((0:(nlat-1)) / nlat * pi) );

%% alloc out array
pmn = zeros(nlat, size(ml, 2));
% Q = single(1j+zeros(size(ml,2),1));

%% FT x
% X = fft(x, nlon, 1); 

%% main loop
i = 1;
for m=0:lmax

    % eq 13
    n = m;
    p0 = amm(m+1)*(1 - cx.*cx).^(m/2)*(-1)^m;
    pmn(:, i)=p0; i=i+1;
    % Q(i)=p0*X(m+1,:)'; i=i+1;
    if n == lmax, break, end

    % eq 14
    n = n + 1;
    amn = sqrt((4*n*n - 1)/(n*n - m*m));
    p1 = amn * cx .* p0;
    pmn(:, i)=p1; i=i+1;
    % Q(i)=p1*X(m+1,:)'; i=i+1;
    if n == lmax, continue, end

    % eq 15 base case
    n = n + 1;
    amn = sqrt((4*n*n - 1)/(n*n - m*m));
    bmn = -sqrt(((2*n + 1)/(2*n - 3))*(((n - 1)*(n - 1) - m*m)/(n*n - m*m)));
    p2 = amn * cx .* p1 + bmn * p0;
    pmn(:, i)=p2; i=i+1;
    % Q(i)=p2*X(m+1,:)'; i=i+1;
    if n == lmax, continue, end

    % eq 15 iterate
    while n < lmax
        p0 = p1;
        p1 = p2;
        n = n + 1;
        amn = sqrt((4*n*n - 1)/(n*n - m*m));
        bmn = -sqrt(((2*n + 1)/(2*n - 3))*(((n - 1)*(n - 1) - m*m)/(n*n - m*m)));
        p2 = amn * cx .* p1 + bmn*p0;
        pmn(:, i)=p2; i=i+1;
        % Q(i)=p2*X(m+1,:)'; i=i+1;
    end

end