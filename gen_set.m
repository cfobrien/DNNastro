% Script for generating a dataset of back projections of images from the
% dataset of ground truths in ./datasets/augmented_dataset_linscale/
% the successful back projections are saved to ./back_projections/

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

if ~exist('../back_projections', 'dir')
    mkdir('../back_projections/');
end

%% Retrieve filenames of ground truths, compute and save BPs to form a dataset
filenames = dir(fullfile('../datasets/augmented_dataset_linscale', '*fits'));

path = pwd;
cd('../matlab_files');

% Iterate over all images in set
for  i = 1 : numel(filenames)
    filename = filenames(i).name
    cd('../matlab_files');

    gt = imresize(im2double(fitsread(filename)),[512 512]);
    Nx = size(gt,1);
    Ny = size(gt,2);
    f = 1.4;
    super_res = 1;%super_res=0; 
    sigma = 0.07;

    % 2. Create the measurement operator and its adjoint
    [A, At, Gw] = generate_data_basic(Nx,Ny,f,super_res,0);

    Phi_t = @(x) HS_forward_operator(x,Gw,A);
    Phi = @(y) HS_adjoint_operator(y,Gw,At,Nx,Ny);

    % Noise addition operator
    add_noise = @(y) (y + (randn(size(y)) + 1i*randn(size(y)))*sigma/sqrt(2));

    % 3. Create the measurements and the back-projection
    y = add_noise(cell2mat(Phi_t(gt)));
    bp = real(Phi({y}));
    bp = rescale(bp);

    % Uncomment to ensure BP looks like it should
    %imshow(bp);

    % Write BP image to BP dataset directory
    fitswrite(bp, ['../back_projections/' filename]);
    %printf("Saved %s\n", filename);

    cd(path);
end

