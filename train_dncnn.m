close all;
clear;
clc;

addpath('../datasets');
addpath('../datasets/augmented_dataset_linscale');

%% DnCNN architecture and implementation

input_size = 64;
num_channels = 32;
layers = [
    imageInputLayer([input_size input_size 1], 'Name','Input')

    convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv11')
    batchNormalizationLayer('Name','bn11')
    reluLayer('Name','relu11')
    
    convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv12')
    batchNormalizationLayer('Name','bn12')
    reluLayer('Name','relu12')
    
    convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv13')
    batchNormalizationLayer('Name','bn13')
    reluLayer('Name','relu13')
    
    convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv14')
    batchNormalizationLayer('Name','bn14')
    reluLayer('Name','relu14')
    
    convolution2dLayer(3,1,'NumChannels',1,'Padding','same','Name','conv15')
    batchNormalizationLayer('Name','bn15')
    reluLayer('Name','relu15')
    
    additionLayer(2,'Name','add_end')
    
    regressionLayer('Name','output') 
];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'Input', 'add_end/in2');

gtds = imageDatastore('../datasets/augmented_dataset_linscale/*.fits', 'ReadFcn', @fitsreadres2double);
noisyds = imageDatastore('../datasets/augmented_dataset_linscale/*.fits', 'ReadFcn', @fits2noisy);
augmenter = imageDataAugmenter('RandXReflection',true, 'RandYReflection',true);
dstrain = randomPatchExtractionDatastore(...
    noisyds, gtds, [64,64], 'DataAugmentation',augmenter);

%% options
%uncomment to use only 2nd gpu
delete(gcp('nocreate'))
parpool('local', numel(1));
gpuDevice(2);

%'LearnRateSchedule', 'piecewise',...
options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-5, ...
    'ExecutionEnvironment','multi-gpu', ...
    'Plots','training-progress', ...
    'Shuffle','every-epoch', ...
    'L2Regularization',0.01, ...
    'MiniBatchSize',8);

%% train and save
%uncomment to continue training previously trained network
load dncnn; lgraph = layerGraph(dncnn);

dncnn = trainNetwork(dstrain, lgraph, options);
save dncnn

%% test net
load dncnn;

%for n = 1 : dstrain.NumObservations
for n = 1 : 15
    test_idx = randi(numel(gtds.Files));
    
    %[data, info] = readByIndex(dstrain,n);
    %noisy = cell2mat(data.InputImage);
    %gt = cell2mat(data.ResponseImage);
    
    gt = normalise(readimage(gtds,test_idx));
    noisy = normalise(readimage(noisyds,test_idx));
    
    res = normalise(cell2mat(compute_net(dncnn,noisy,64)));
    
    figure
    subplot(1,3,1);
    imshow(gt);
    title('Ground truth');
    subplot(1,3,2);
    imshow(noisy);
    title('Image with Gaussian AWN');
    subplot(1,3,3);
    imshow(res);
    impixelinfo;
    rsnr = 20*log10(norm(gt(:))/norm(gt(:)-res(:)));
    title(strcat('Denoising SNR: ', num2str(rsnr), ' dB'));
    if ~exist('../results', 'dir')
        mkdir('../results/');
    end
    saveas(gcf, ['../results/net_test_' mat2str(test_idx)]);
end


%% functions
function A_norm = normalise(A)
    A_norm = (A-min(A(:))) ./ (max(A(:)-min(A(:))));
    A_norm(isnan(A_norm)) = 0;
end

function im = fitsreadres2double(file)
    im = normalise(imresize(fitsread(file), [512 512]));
end

function out = fits2noisy(file)
    % Noise addition operator
    sigma = 0.07;
    add_noise_to_picture = @(x) x + sigma*randn(size(x));
    out = fitsreadres2double(file);
    out = add_noise_to_picture(out);
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

function dataOut = augmentImages(data)

dataOut = cell(size(data));
    for idx = 1:size(data,1)
        rot90Val = randi(4,1,1)-1;
        dataOut(idx,:) = {rot90(data{idx,1},rot90Val),rot90(data{idx,2},rot90Val)};
    end
end
