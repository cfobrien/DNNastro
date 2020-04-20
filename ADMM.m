close all;
clear;
clc;

%% Setting up paths
addpath('../matlab_files/');
addpath('../matlab_files/utils/');
addpath('../matlab_files/utils/lib/');
addpath('../matlab_files/samples');
addpath('../matlab_files/simulated_data');
addpath('../matlab_files/simulated_data/data');
addpath('../datasets');
addpath('../datasets/augmented_dataset_linscale');
run('../matlab_files/utils/lib/irt/setup.m');

if ~exist('../ADMM_reconstructions', 'dir')
    mkdir('../ADMM_reconstructions/');
end

%% Retrieve filenames of ground truths, compute and save BPs to form a dataset
filenames = dir(fullfile('../datasets/augmented_dataset_linscale', '*fits'));

nlevel = 4;
wv='db4';
dwtmode('per');

path = pwd;

Nx = 512;
Ny = 512;
f = 1.4;
super_res = 1;
sigma = 0.07;

% Noise addition operator
add_noise = @(y) (y + (randn(size(y)) + 1i*randn(size(y)))*sigma/sqrt(2));

arr_snr = zeros(50, 1);

for i = 1 : 50 %numel(filenames)
    
    filename = filenames(i).name;
    cd('../matlab_files');

    clear A
    clear At
    clear Gw
    clear Phi_t
    clear Phi
    
    % Create the measurement operator and its adjoint
    [A, At, Gw] = generate_data_basic(Nx,Ny,f,super_res,0);
    Phi_t = @(x) HS_forward_operator(x,Gw,A);
    Phi = @(y) HS_adjoint_operator(y,Gw,At,Nx,Ny);
       
    gt = imresize((normalise(fitsread(filename))),[512 512]);
    max_gt = max(gt(:));
    y = cell2mat(Phi_t(gt));
    bp = normalise(real(Phi({add_noise(y)})));
    bp = bp .* max_gt;
    
    [alphad,S]=wavedec2(bp,nlevel,wv);
    Psi_t = @(x) wavedec2(x,nlevel,wv); 
    Psi = @(x) waverec2(x,S,wv);
    
    f = @(x) sum(abs(Psi_t(x)));
    %f = @(x) norm(y - Phit(x), 2)^2;
    
    shrink = @(z, d) max(abs(z)-d, 0).*sign(z);
    vt = zeros(size(y));
    st = -y;
    xt = ones(512);

    M = 0.11*Nx*Ny;
    epsilon = sigma * sqrt(M + 2*sqrt(M));
    
    sc = @(z) z*min(epsilon/norm(z(:)), 1);
    nt = sc(st);
    delta = 1e-7;
    rho = 1e-7; %threshold value
    rel_tol = 1e-2;
    rel_tol2 = 1e-2;
    max_iter = 10;
    fprev = 0;
    xprev = zeros(512);
    t = 1;
    fval = 9999;

    %tstart=tic;
    converged = false;
    
    while(~converged)
        converged = (abs((fval - fprev)/fval) < rel_tol) & (norm(y - cell2mat(Phi_t(xt)), 2) <= epsilon);
        converged = converged | (norm(xt(:) - xprev(:), 2)/norm(xt(:), 2) <= rel_tol2);
        converged = converged | (t + 1 >= max_iter);
        fprev = fval;
        xprev = xt;
        xt = Psi(shrink(Psi_t(xt - delta*real(Phi({st + nt - vt}))), delta * rho^(-1)));
        %test1 = delta*real(Phi(st + nt - vt));
        %test2 = Psit(xt - test1);
        %test3 = shrink(test2, delta * rho^(-1));
        %xt = Psi(test3);
        st = cell2mat(Phi_t(xt)) - y;
        nt = sc(vt - st);
        vt = vt - (st + nt);
        t = t + 1;
        fval = f(xt);
        %fprintf("%e\n", (abs((fval - fprev)/fval) < rel_tol));
    end
    
    rsnr = 20*log10(norm(gt(:))/norm(gt(:)-xt(:)));
    rsnr_bp = 20*log10(norm(gt(:))/norm(gt(:)-bp(:)));
    result = cat(2,gt,bp,xt);
    imshow(result);
    fprintf("Reconstructed in %d iterations; bp snr: %d, snr: %d\n", t, rsnr_bp, rsnr);
    arr_snr(i) = rsnr;
    
    cd(path);
end

function A_norm = normalise(A)
    A_norm = (A-min(A(:))) ./ (max(A(:)-min(A(:))));
    A_norm(isnan(A_norm)) = 0;
end
