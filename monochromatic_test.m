%% Operators for monochromatic Radio-Interferometry
% Setup and initialization
clc;
clear;
close all;

addpath('../matlab_files/utils/');
addpath('../matlab_files/utils/lib/');
addpath('../matlab_files/samples');
addpath('../matlab_files/simulated_data');
addpath('../matlab_files/simulated_data/data');

run('../matlab_files/utils/lib/irt/setup.m');

GEN_SET = 0;
TRAIN_NET = 0;
TEST_NET = 1;

% Adjointness check
% u = randn(Nx,Ny);
% y1_cell = Phi_t(u);
% y1 = y1_cell{1};
% v = randn(size(y1));
% v_adj = Phi({v});
% norm(real(v'*y1-v_adj(:)'*u(:)))

%% Setup NN
input_size = 256;
num_channels = 32;

layers = [
    imageInputLayer([input_size input_size 1], 'Name','Input')
    
    %Depth1
    convolution2dLayer(3,num_channels,'Padding','same','Name','conv11')
    batchNormalizationLayer('Name','bn11')
    reluLayer('Name','relu11')
    
    convolution2dLayer(3,num_channels,'Padding','same','Name','conv12')
    batchNormalizationLayer('Name','bn12')
    reluLayer('Name','relu12')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MP1')
    
    %Depth2
    convolution2dLayer(3,num_channels*2,'Padding','same','Name','conv21')
    batchNormalizationLayer('Name','bn21')
    reluLayer('Name','relu21')
    
    convolution2dLayer(3,num_channels*2,'Padding','same','Name','conv22')
    batchNormalizationLayer('Name','bn22')
    reluLayer('Name','relu22')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MP2')
    
    %Depth3
    convolution2dLayer(3,num_channels*4,'Padding','same','Name','conv31')
    batchNormalizationLayer('Name','bn31')
    reluLayer('Name','relu31')
    
    convolution2dLayer(3,num_channels*4,'Padding','same','Name','conv32')
    batchNormalizationLayer('Name','bn32')
    reluLayer('Name','relu32')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MP3')
    
    %Depth4
    convolution2dLayer(3,num_channels*8,'Padding','same','Name','conv41')
    batchNormalizationLayer('Name','bn41')
    reluLayer('Name','relu41')
    
    convolution2dLayer(3,num_channels*8,'Padding','same','Name','conv42')
    batchNormalizationLayer('Name','bn42')
    reluLayer('Name','relu42')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MP4')
    
    %Depth5
    convolution2dLayer(3,num_channels*16,'Padding','same','Name','conv51')
    batchNormalizationLayer('Name','bn51')
    reluLayer('Name','relu51')
    
    convolution2dLayer(3,num_channels*16,'Padding','same','Name','conv52')
    batchNormalizationLayer('Name','bn52')
    reluLayer('Name','relu52')
    
    transposedConv2dLayer(2,num_channels*8,'Stride',2,'Name','up53')
    batchNormalizationLayer('Name','bn53')
    reluLayer('Name','relu53')
    
    %up-Depth4
    depthConcatenationLayer(2,'Name','concat4')
    convolution2dLayer(3,num_channels*8,'Padding','same','Name','conv43')
    batchNormalizationLayer('Name','bn43')
    reluLayer('Name','relu43')
    
    convolution2dLayer(3,num_channels*8,'Padding','same','Name','con44')
    batchNormalizationLayer('Name','bn44')
    reluLayer('Name','relu44')
    
    transposedConv2dLayer(2,num_channels*4,'Stride',2,'Name','up45')
    batchNormalizationLayer('Name','bn45')
    reluLayer('Name','relu45')
    
    %up-Depth3
    depthConcatenationLayer(2,'Name','concat3')
    convolution2dLayer(3,num_channels*4,'Padding','same','Name','conv33')
    batchNormalizationLayer('Name','bn33')
    reluLayer('Name','relu33')
    
    convolution2dLayer(3,num_channels*4,'Padding','same','Name','conv34')
    batchNormalizationLayer('Name','bn34')
    reluLayer('Name','relu34')
    
    transposedConv2dLayer(2,num_channels*2,'Stride',2,'Name','up35')
    batchNormalizationLayer('Name','bn35')
    reluLayer('Name','relu35')
    
    %up-Depth2
    depthConcatenationLayer(2,'Name','concat2')
    convolution2dLayer(3,num_channels*2,'Padding','same','Name','conv23')
    batchNormalizationLayer('Name','bn23')
    reluLayer('Name','relu23')
    
    convolution2dLayer(3,num_channels*2,'Padding','same','Name','conv24')
    batchNormalizationLayer('Name','bn24')
    reluLayer('Name','relu24')
    
    transposedConv2dLayer(2,num_channels,'Stride',2,'Name','up25')
    batchNormalizationLayer('Name','bn25')
    reluLayer('Name','relu25')
    
    %up-Depth1
    depthConcatenationLayer(2,'Name','concat1')
    convolution2dLayer(3,num_channels,'Padding','same','Name','conv13')
    batchNormalizationLayer('Name','bn13')
    reluLayer('Name','relu13')
    
    convolution2dLayer(3,num_channels,'Padding','same','Name','conv14')
    batchNormalizationLayer('Name','bn14')
    reluLayer('Name','relu14')
    
    convolution2dLayer(3,1,'Padding','same','Name','conv15')
    batchNormalizationLayer('Name','bn15')
    reluLayer('Name','relu15')
    
    additionLayer(2,'Name','add_end')
    
    regressionLayer('Name','output')

];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'relu42', 'concat4/in2');
lgraph = connectLayers(lgraph, 'relu32', 'concat3/in2');
lgraph = connectLayers(lgraph, 'relu22', 'concat2/in2');
lgraph = connectLayers(lgraph, 'relu12', 'concat1/in2');
lgraph = connectLayers(lgraph, 'Input', 'add_end/in2');

analyzeNetwork(lgraph);

%% Generate back projected set
if (GEN_SET)
    if ~exist('../back_projections', 'dir')
        mkdir('../', 'back_projections');
    end

    if ~exist('../datasets/augmented_dataset_linscale/success', 'dir')
        mkdir('../', 'datasets/augmented_dataset_linscale/success');
    end

    if ~exist('../datasets/augmented_dataset_linscale/failed', 'dir')
        mkdir('../datasets/augmented_dataset_linscale/', 'failed');
    end

    addpath('../datasets');
    addpath('../datasets/augmented_dataset_linscale');
    path = pwd;

    filenames = dir(fullfile('../datasets/augmented_dataset_linscale', '*fits'));
    %for i = 1 : numel(filenames)
    for i = 1 : 200
        try
            filename = filenames(i).name;
            im = get_bp(['../datasets/augmented_dataset_linscale/' filename]);

            %gt = fitsread(['../datasets/augmented_dataset_linscale/' filename]);
            %y = Phi_t(gt);
            %im = real(Phi(y));
            %im = im/max(bproj(:));

            fitswrite(im, ['../back_projections/' filename]);

            cd('../datasets/augmented_dataset_linscale/');
            copyfile(filename, 'success');
            cd(path);

            %fprintf('Saved image %s to %s\n', filenames(i).name, '../back_projections/');
        catch
            warning('failed to get backprojection, ignoring file in dataset');
            cd('../datasets/augmented_dataset_linscale/');
            movefile(filename, 'failed');
            cd(path);
        end
    end
end

%% Setput datastores, train and save net
if (TRAIN_NET)
    augmenter = imageDataAugmenter( ...
        'RandRotation',@()randi([0,1],1)*90, ...
        'RandXReflection',true);

    patchSize = input_size;%[input_size input_size];
    patchds = randomPatchExtractionDatastore( ...
        imageDatastore('../datasets/augmented_dataset_linscale/success/*.fits', 'ReadFcn', @fitsread), ...
        imageDatastore('../back_projections/*.fits', 'ReadFcn', @fitsread), ...
        patchSize, 'PatchesPerImage',1, 'DataAugmentation',augmenter);

    %shuffle?

    options = trainingOptions('adam', ...
        'MaxEpochs',130,...
        'ExecutionEnvironment','multi-gpu', ...
        'InitialLearnRate',1e-3, ...
        'Verbose',false, ...
        'Plots','training-progress', ...
        'Shuffle', 'every-epoch', ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 40, ...
        'LearnRateDropFactor', 0.1, ...
        'GradientThreshold', 2, ...
        'MiniBatchSize',7);

    net = trainNetwork(patchds, lgraph, options);
    trainednet1 = net;
    save trainednet1
end

%% Test net
if (TEST_NET)
    gt = fitsread('../datasets/augmented_dataset_linscale/gen_groundtruth_10.fits');
    %bp = fitsread('../back_projections/gen_groundtruth_0.fits');
    bp = get_bp('../datasets/augmented_dataset_linscale/gen_groundtruth_10.fits');
    load trainednet1;
    res = cell2mat(compute_net(trainednet1, bp, input_size));
    figure
    subplot(1,3,1);
    imshow(gt);
    title('Ground truth');
    subplot(1,3,2);
    imshow(bp);
    title('Back projection');
    subplot(1,3,3);
    imshow(res);
    title('Reconstruction');
    
end

%% Get backprojection
function bproj = get_bp(file)
    gtr = fitsread(file);
    
    Nx = size(gtr,1);
    Ny = size(gtr,2);
    f = 1.4;
    super_res=0; 

    cd('../matlab_files');
    % 2. Create the measurement operator and its adjoint
    [A, At, Gw] = generate_data_basic(Nx,Ny,f,super_res,0);

    Phi_t = @(x) HS_forward_operator(x,Gw,A);
    Phi = @(y) HS_adjoint_operator(y,Gw,At,Nx,Ny);

    cd('../DNNastro');
    % 3. Create the measurements and the back-projection
    y = Phi_t(gtr);
    bproj = real(Phi(y));
    bproj = bproj/max(bproj(:));
end

%% Compute from patches
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