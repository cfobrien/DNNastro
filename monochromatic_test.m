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
    
    % Depth 1
    convolution2dLayer(3,num_channels,'Padding','same','Name','conv11')
    batchNormalizationLayer('Name','bn11')
    reluLayer('Name','relu11')

    convolution2dLayer(3,num_channels,'Padding','same','Name','conv12')
    batchNormalizationLayer('Name','bn12')
    reluLayer('Name','relu12')

    convolution2dLayer(3,num_channels,'Padding','same','Name','conv13')
    batchNormalizationLayer('Name','bn13')
    reluLayer('Name','relu13')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MP1')
    
    % Depth 2
    convolution2dLayer(3,num_channels*2,'Padding','same','Name','conv21')
    batchNormalizationLayer('Name','bn21')
    reluLayer('Name','relu21')
    
    convolution2dLayer(3,num_channels*2,'Padding','same','Name','conv22')
    batchNormalizationLayer('Name','bn22')
    reluLayer('Name','relu22')
    
    maxPooling2dLayer(2,'Stride',2, 'Name','MP2')
    
    % Depth 3
    convolution2dLayer(3,num_channels*4,'Padding','same','Name','conv31')
    batchNormalizationLayer('Name','bn31')
    reluLayer('Name','relu31')
    
    convolution2dLayer(3,num_channels*4,'Padding','same','Name','conv32')
    batchNormalizationLayer('Name','bn32')
    reluLayer('Name','relu32')
    
    transposedConv2dLayer(2,num_channels*2,'Stride',2,'Name','up32')
    batchNormalizationLayer('Name','bn33')
    reluLayer('Name','relu33')
     
    % Back to depth 2
    depthConcatenationLayer(2,'Name','concat2')
    convolution2dLayer(3,num_channels*2,'Padding','same','Name','conv23')
    batchNormalizationLayer('Name','bn23')
    reluLayer('Name','relu23')
    
    convolution2dLayer(3,num_channels*2,'Padding','same','Name','conv24')
    batchNormalizationLayer('Name','bn24')
    reluLayer('Name','relu24')
    
    transposedConv2dLayer(2,num_channels*2,'Stride',2,'Name','up25')
    batchNormalizationLayer('Name','bn25')
    reluLayer('Name','relu25')
    
    % Back to depth 1
    depthConcatenationLayer(2,'Name','concat1')
    convolution2dLayer(3,num_channels,'Padding','same','Name','conv14')
    batchNormalizationLayer('Name','bn14')
    reluLayer('Name','relu14')
    
    convolution2dLayer(3,1,'Padding','same','Name','conv15')
    batchNormalizationLayer('Name','bn15')
    reluLayer('Name','relu15')
    
    convolution2dLayer(3,1,'Padding','same','Name','conv16')
    batchNormalizationLayer('Name','bn16')
    reluLayer('Name','relu16')
    
    additionLayer(2,'Name','add_end')
    
    regressionLayer('Name','output')
];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'relu22', 'concat2/in2');
lgraph = connectLayers(lgraph, 'relu13', 'concat1/in2');
lgraph = connectLayers(lgraph, 'Input', 'add_end/in2');

%analyzeNetwork(lgraph);

%% Generate back projected set
if ~exist('../back_projections', 'dir')
    mkdir('../', 'back_projections');
end

filenames = dir(fullfile('../matlab_files/samples', '*fits'));
for i = 1 : numel(filenames) -1 % Temp fix, -1 since last fits file seems corrupted
    %im = get_bp(['../matlab_files/samples/' filenames(i).name]);
    %imshow(im);
    fitswrite(get_bp(['../matlab_files/samples/' filenames(i).name]), ['../back_projections/' filenames(i).name]);
    fprintf('Saved image %s to %s\n', filenames(i).name, '../back_projections/');
end

%% Setup patch extraction datastore
augmenter = imageDataAugmenter( ...
    'RandRotation',@()randi([0,1],1)*90, ...
    'RandXReflection',true);

patchSize = input_size;%[input_size input_size];
patchds = randomPatchExtractionDatastore( ...
    imageDatastore('../matlab_files/samples/*.fits', 'ReadFcn', @fitsread), ...
    imageDatastore('../back_projections/*.fits', 'ReadFcn', @fitsread), ...
    patchSize, 'PatchesPerImage',1, 'DataAugmentation',augmenter);

%shuffle?

%% Train and save net
options = trainingOptions('adam', ...
    'MaxEpochs',130,...
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
save supernet1


%% Get backprojection
function gtruth = get_gt(file)
    gtruth = fitsread(file);
end

function bproj = get_bp(file)
    gtr = get_gt(file);
    
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
