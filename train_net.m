close all;
clear;
clc;

%% layers
input_size = 512;
num_channels = 64;

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

%analyzeNetwork(lgraph);


%% datastores
%gtds = imageDatastore('../preprocessed/gt/*.fits', 'ReadFcn', @fitsreadres2double);
gtds = imageDatastore('../datasets/augmented_dataset_linscale/*.fits', 'ReadFcn', @fitsreadres2double);
%bpds = imageDatastore('../preprocessed/bp/*.fits', 'ReadFcn', @fitsreadres2double);
bpds = imageDatastore('../back_projections/*.fits', 'ReadFcn', @fitsreadres2double);

%augmenter = imageDataAugmenter('RandRotation',[0 90],'RandXReflection',true);
augmenter = imageDataAugmenter('RandXReflection',true, 'RandYReflection',true);
rpeds = randomPatchExtractionDatastore(bpds, gtds, [512, 512], 'PatchesPerImage',1, 'DataAugmentation',augmenter);
rpeds = shuffle(rpeds);

%Split datastore
%[dstrain, dstest] = splitEachLabel(rpeds, 0.9, 'randomized');

LAST_IDX = rpeds.NumObservations;
SPLIT_IDX = LAST_IDX * 0.8;

dstrain = partitionByIndex(rpeds, 1 : SPLIT_IDX);
dstest = partitionByIndex(rpeds, SPLIT_IDX+1 : LAST_IDX);
dstest.MiniBatchSize = 1;


%% options
%uncomment to use only 2nd gpu
% delete(gcp('nocreate'))
% parpool('local', numel(1));
% gpuDevice(2);

options = trainingOptions('adam', ...
    'MaxEpochs',15, ...
    'InitialLearnRate',1e-5, ...
    'ExecutionEnvironment','multi-gpu', ...
    'Plots','training-progress', ...
    'Shuffle','every-epoch', ...
    'MiniBatchSize',8);
%'L2Regularization',0.00001, ...

%% train and save
%uncomment to continue training previously trained network
load net; lgraph = layerGraph(net);

net = trainNetwork(dstrain, lgraph, options);
save net

%% test net
load net;
%for n = 1 : dstest.NumObservations
for n = 1 : 15
    test_idx = randi(numel(gtds.Files));
    
    [data, info] = readByIndex(dstest,n);
    gt = cell2mat(data.ResponseImage);
    bp = cell2mat(data.InputImage);

    max_gt = max(gt(:));
    bp = bp .* max_gt;
    
    res = normalise(predict(net,bp));
    
    figure
    subplot(1,3,1);
    imshow(gt);
    title('Ground truth');
    subplot(1,3,2);
    imshow(bp);
    title('Back projection');
    subplot(1,3,3);
    imshow(res);
    impixelinfo;
    rsnr = 20*log10(norm(gt(:))/norm(gt(:)-res(:)));
    title(strcat('Reconstruction SNR: ', num2str(rsnr), ' dB'));
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
