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

if ~exist('../hybrid_reconstructions', 'dir')
    mkdir('../hybrid_reconstructions/');
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

file = fopen('../results/hybrid_results.txt', 'wt');

numiter = 5;%numel(filenames);
arr_snr = zeros(numiter, 1);
arr_ext = zeros(numiter, 1);

load dncnn

for i = 1 : numiter
    
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
    
%     [alphad,S]=wavedec2(bp,nlevel,wv);
%     Psi_t = @(x) wavedec2(x,nlevel,wv); 
%     Psi = @(x) waverec2(x,S,wv);
    
%     f = @(x) sum(abs(Psi_t(x)));
    %f = @(x) norm(y - Phit(x), 2)^2;
    
%     shrink = @(z, d) max(abs(z)-d, 0).*sign(z);
    vt = zeros(size(y));
    st = -y;
    xt = ones(512);

    M = 0.11*Nx*Ny;
    epsilon = sigma * sqrt(M + 2*sqrt(M));
    
    sc = @(z) z*min(epsilon/norm(z(:)), 1);
    nt = sc(st);
    delta = 1e-7;
    %rho = 1e-7; %threshold value
    %rel_tol = 1e-2;
    rel_tol2 = 1e-5;
    max_iter = 1000;
    fprev = 0;
    xprev = zeros(512);
    t = 1;
    fval = 9999;

    tstart=tic;
    converged = false;
    
    while(~converged)
        %converged = (abs((fval - fprev)/fval) < rel_tol) & (norm(y - cell2mat(Phi_t(xt)), 2) <= epsilon);
        converged = converged | (norm(xt(:) - xprev(:), 2)/norm(xt(:), 2) <= rel_tol2);
        converged = converged | (t + 1 >= max_iter);
        %fprev = fval;
        xprev = xt;
        %xt = Psi(shrink(Psi_t(xt - delta*real(Phi({st + nt - vt}))), delta * rho^(-1)));
        
        xt = normalise(xt);
        difference = xt - delta*real(Phi({st + nt - vt}));
        
        xt = im2double(cell2mat(compute_net(dncnn, difference, 64)));
        
        st = cell2mat(Phi_t(xt)) - y;
        nt = sc(vt - st);
        vt = vt - (st + nt);
        t = t + 1;
        %fval = f(xt);
        %fprintf("%e\n", (abs((fval - fprev)/fval) < rel_tol));
        imshow(xt);
    end
    tend=toc(tstart);
    rsnr = 20*log10(norm(gt(:))/norm(gt(:)-xt(:)));
    rsnr_bp = 20*log10(norm(gt(:))/norm(gt(:)-bp(:)));
    result = cat(2,gt,bp,xt);
    imshow(result);
    %fprintf("Reconstructed in %d iterations; bp snr: %d, snr: %d\n", t, rsnr_bp, rsnr);
    fprintf(file, "SNR: %e Reconstruction Time: %es (%d iterations)\n", rsnr, tend, t);
    imwrite(xt, ['../hybrid_reconstructions/' filename], 'png');
    arr_snr(i) = rsnr;
    arr_ext(i) = tend;
    cd(path);
end

snr_stdev = std(arr_snr);
ext_stdev = std(arr_ext);
avg_snr = sum(arr_snr)/numel(arr_snr);
avg_ext = sum(arr_ext)/numel(arr_ext);
fprintf(file, "\nAverage SNR = %e, Standard Deviation = %e\n ", avg_snr, snr_stdev);
fprintf(file, "\nAverage ex. time = %e, Standard Deviation = %e\n ", avg_ext, ext_stdev);

function A_norm = normalise(A)
    A_norm = (A-min(A(:))) ./ (max(A(:)-min(A(:))));
    A_norm(isnan(A_norm)) = 0;
end

function o = compute_net(net,I,n)
% This function takes as input:
%   - net: a neural network
%   - I: an image
%   - n: a patchsize
% It returns:
%   - o: a picture built by the application of the network on patches of n
%   by n of the full image I.

imSz = size(I);
patchSz = [n n];
xIdxs = [1:patchSz(2):imSz(2) imSz(2)+1]; % [16 64+1]
yIdxs = [1:patchSz(1):imSz(1) imSz(1)+1];
patches = cell(length(yIdxs)-1,length(xIdxs)-1);
for i = 1:length(yIdxs)-1
    Isub = I(yIdxs(i):yIdxs(i+1)-1,:);
    for j = 1:length(xIdxs)-1
        sub_picture = Isub(:,xIdxs(j):xIdxs(j+1)-1);
        % Here compute the output of the network on each patch
        if size(sub_picture) == patchSz
            patches{i,j} = predict(net,sub_picture);
        end
    end
end
o = patches;
end
