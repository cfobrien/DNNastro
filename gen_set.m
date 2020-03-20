% Script for generating a dataset of back projections of images from the
% dataset of ground truths in ./datasets/augmented_dataset_linscale/
% the successful back projections are saved to ./back_projections/

clear all; close all; clc;


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



%% Create directory to store all ground truths that can't be back projected
% These must be removed from since GT & BP datasets must be of the same size
if ~exist('../datasets/augmented_dataset_linscale/failed', 'dir')
    mkdir('../datasets/augmented_dataset_linscale/', 'failed');
end



%% Retrieve filenames of ground truths, compute and save BPs to form a dataset
filenames = dir(fullfile('../datasets/augmented_dataset_linscale', '*fits'));

path = pwd;
cd('../matlab_files');

% Iterate over all images in set
parfor  i = 1 : numel(filenames)
    filename = filenames(i).name;
    cd('../matlab_files');
    
     try
        gt = im2double(fitsread(filename));
        Nx = size(gt,1);
        Ny = size(gt,2);
        f = 1.4;
        super_res=0; 

        % 2. Create the measurement operator and its adjoint
        [A, At, Gw] = generate_data_basic(Nx,Ny,f,super_res,0);

        Phi_t = @(x) HS_forward_operator(x,Gw,A);
        Phi = @(y) HS_adjoint_operator(y,Gw,At,Nx,Ny);

        % 3. Create the measurements and the back-projection
        y = Phi_t(gt);
        bp = real(Phi(y));
        bp = rescale(bp);
        
        % Uncomment to ensure BP looks like it should
        %imshow(bp);
        
        % Write BP image to BP dataset directory
        fitswrite(bp, ['../back_projections/' filename]);
        %printf("Saved %s\n", filename);
    catch
        %warning(['Failed to get back projection for image ' filename]);
        cd('../datasets/augmented_dataset_linscale');
        copyfile(filename, 'failed');
        delete(filename);
        cd('../../matlab_files');
    end

        cd(path);
end

