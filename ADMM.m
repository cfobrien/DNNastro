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

%Nx = size(gt,1);
%Ny = size(gt,2);
Nx = 512;
Ny = 512;
f = 1.4;
super_res = 1;%super_res=0; 
sigma = 30;

% Noise addition operator
add_noise = @(y) (y + (randn(size(y)) + 1i*randn(size(y)))*sigma/sqrt(2));

% Iterate over all images in set
for  i = 1 : 1%numel(filenames)
    filename = filenames(i).name;
    cd('../matlab_files');

    gt = fitsread(filename);
    max_gt = max(max(gt));
    
    gt = imresize((normalise(gt)),[512 512]);
    
    clear A; clear At; clear Gw;
    clear Phi_t;
    clear Phi;
    
    % 2. Create the measurement operator and its adjoint
    [A, At, Gw] = generate_data_basic(Nx,Ny,f,super_res,0);
    
    Phi_t = @(x) HS_forward_operator(x,Gw,A);
    Phi = @(y) HS_adjoint_operator(y,Gw,At,Nx,Ny);

    
    
    % 3. Create the measurements and the back-projection
    y = cell2mat(Phi_t(gt));
    bp = real(Phi({add_noise(y)}));
    bp = normalise(bp);
    bp = bp .* max_gt;
    
    [alphad,S]=wavedec2(bp,nlevel,wv);
    Psit = @(x) wavedec2(x, nlevel,wv); 
    Psi = @(x) waverec2(x,S,wv);

    %setup convergence loop
    f = @(x) sum(abs(Psit(x)));
    %f = @(x) norm(y - Phit(x), 2)^2;

    shrink = @(z, d) max(abs(z)-d, 0).*sign(z);
    vt = zeros(size(y));
    st = -y;
    xt = ones(512);

    
    %epsilon = sigma * sqrt(M + 2*sqrt(M));
    epsilon = sigma * sqrt(size(y) + 2 * sqrt(size(y)));
    
    sc = @(z) z*min(epsilon/norm(z(:)), 1);
    nt = sc(st);
    delta = 1e0;
    rho = 1e2; %threshold value
    rel_tol = 1e-4;
    rel_tol2 = 1e-5;
    max_iter = 100;
    fprev = 0;
    xprev = zeros(512);
    t = 1;
    fval = 9999;

    %tstart=tic;
    converged = false;

    while(~converged)
            %converged = (abs((fval - fprev)/fval) < rel_tol) & (norm(y - Phit(xt), 2) <= epsilon);
            %converged = converged | (norm(xt(:) - xprev(:), 2)/norm(xt(:), 2) <= rel_tol2);
            converged = converged | (t + 1 >= max_iter);
            fprev = fval;
            xprev = xt;
            xt = Psi(shrink(Psit(xt - delta*normalise(real(Phi({st + nt - vt})))), delta * rho^(-1)));
            test1 = Phi_t(xt);
            test2 = cell2mat(test1);
            st = test2 - y;
            %st = cell2mat(Phit(xt)) - y;
            nt = sc(vt - st);
            vt = vt - (st + nt);
            t = t + 1;
            fval = f(xt);
            %fprintf("%e\n", (abs((fval - fprev)/fval) < rel_tol));
    end
    %gt = imread(['../datasets/augmented_dataset_linscale/test/' filenames_gt(i).name]);
    result = cat(2,gt,bp,xt);
    imshow(result);
    
    %fitswrite(xt, ['../DNNastro/ADMM_reconstructions/' filename]);
    f = fullfile(path,'ADMM_reconstructions',filename);
    fitswrite(xt, f);
    snr = 20*log10(norm(gt(:))/norm(gt(:)-xt(:)));
    fprintf(file, "%s : SNR: %e\n", filename, snr);
    imshow(result);
    
    
    cd(path);
end

function A = normalise(A)
    A = (A-min(A(:))) ./ (max(A(:)-min(A(:))));
end