%%ADMM test
%

close all;
clear;
clc;

addpath('../matlab_files/');
addpath('../matlab_files/utils/');
addpath('../matlab_files/utils/lib/');
addpath('../matlab_files/samples');
addpath('../matlab_files/simulated_data');
addpath('../matlab_files/simulated_data/data');

run('../matlab_files/utils/lib/irt/setup.m');

% Definition of the sparsity operators Psi and Psit
nlevel = 4;
wv='db4';
dwtmode('per');

%filenames_gt = dir(fullfile('../datasets/augmented_dataset_linscale/test', '*.fits'));

n = 512;
N1 = 512;
N2 = 512;
N = N1*N2;
f = 1.4;

% Definition of the mask
num_meas = 60;              % Number of measurements
M = num_meas*N2;
w = 10;                     % Width of the low frequency selected band
num_meas = num_meas-w;

sigma = 0.07;

filenames = dir(fullfile('../datasets/augmented_dataset_linscale/', '*.fits'));
nb_files = numel(filenames);
file = fopen('../DNNastro/ADMMresults.txt', 'wt');

% Noise addition operator
add_noise = @(y) (y + (randn(size(y)) + 1i*randn(size(y)))*sigma/sqrt(2));

for i = 1 : 9
    filename = filenames(i).name;
    
    %from new 
    super_res = 1;
    [A, At, Gw] = generate_data_basic(N1,N2,f,super_res,0);
    Phit = @(x) HS_forward_operator(x,Gw,A);
    Phi = @(y) HS_adjoint_operator(y,Gw,At,N1,N2);
    
    %try
        %gt = imresize(im2double(fitsread(filename)),[512 512]);
        %y = Phit(gt);
        %bp = real(Phi(add_noise(y)));
        %from new
        gt = imresize(im2double(fitsread(filename)),[512 512]);
        max_gt = max(max(gt))
        printf("Im trying to read: ", filename);
        y = cell2mat(Phit(gt));
        bp = real(Phi({add_noise(y)}));
        bp = im2double(bp);
        bp = bp.*max_gt;
        %xd = cell2mat(bp);

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

        epsilon = sigma * sqrt(M + 2*sqrt(M));

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
                xt = Psi(shrink(Psit(xt - delta*im2double(real(Phi({st + nt - vt})))), delta * rho^(-1)));
                test1 = Phit(xt);
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
        fitswrite(result, ['../DNNastro/ADMMresults/' filename]);
        snr = 20*log10(norm(gt(:))/norm(gt(:)-xt(:)));
        fprintf(file, "%s : SNR: %e\n", filename, snr);
        imshow(result);
    %catch
        %warning('failed to get backprojection, ignoring file in dataset');
        %cd(path);
        %printf("here\n");
    %end
end