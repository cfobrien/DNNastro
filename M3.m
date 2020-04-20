%% M3.m file
%
% %
% This file is for your implementation and validation of M3 (sections 6 and 7).
% It contains useful functions (sections 1 to 4) as well as a partial
% implementation of the DnCNN for you to finalise (section 5).  

%% 1. Fourier measurements and noise operators

n = 256;
N1 = 256;
N2 = 256;
N = N1*N2;
% Given an image im
%n = size(im,1);
%N1 = size(im,1);
%N2 = size(im,2);
%N = N1*N2;

% Definition of the mask
num_meas = 60;              % Number of measurements
M = num_meas*N2;
w = 10;                     % Width of the low frequency selected band
num_meas = num_meas-w;

ind_mask = randi(n,[num_meas,1]);
mask = zeros(n,n);
mask(floor(n/2-w):floor(n/2+w),:) = 1;

mask(ind_mask,:) = 1;
mask(1,:) = 0;
mask(n,:) = 0;

% Definition of the measurement operators Phit and Phi
Phit = @(x) reshape(ifftshift(mask.*fftshift(fft2(x))),N,1)/sqrt(N);
Phi = @(x) real(ifft2(ifftshift(mask.*fftshift(reshape(x,N1,N2)))))*sqrt(N);

% Noise addition operator
sigma = 0.07;
add_noise = @(y) y + (randn(size(y)) + 1i*randn(size(y)))*sigma/sqrt(2);


%% 3. Noise addition to the training and testing datasets
% You need to add noise to the groundtruth images to create the input
% to the denoising network. This can be achieved with the following
% function.
sigma = 0.07;
add_noise_to_picture = @(x) x + sigma*randn(size(x));



%% 5. DnCNN architecture and implementation

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



%% 6. M3 implementation ...
% Use the functions provided above and your completed Unet architecture as
% appropriate

NET_TRAINED = true; %check to see if we want to retrain net or load previous state

if (~NET_TRAINED)
    
    augmenter = imageDataAugmenter('RandRotation',[0 90],'RandXReflection',true);
    combined_set = randomPatchExtractionDatastore(imageDatastore('C:\Users\lewis\Desktop\CW1/trainingset/*.png', 'ReadFcn', @NormalizeImageResize), ...
        imageDatastore('C:\Users\lewis\Desktop\CW1/trainingset/*.png', 'ReadFcn', @addNoiseResize), ...
        [64,64], 'DataAugmentation',augmenter);
    

    options = trainingOptions('adam', ...
        'ExecutionEnvironment','gpu', ...
        'LearnRateSchedule', 'piecewise',...
        'MaxEpochs',10,...
        'InitialLearnRate',1e-4, ...
        'Verbose',false, ...
        'Plots','training-progress', ...
        'MiniBatchSize',7, ...
        'Shuffle', 'never');


    net = trainNetwork(combined_set, lgraph, options);
    
    trainednet1 = net;
    save m3trainednet7mb10epochs
end

delta = 1e0;
rel_tol = 1e-5;
rel_tol2 = 1e-5;
epsilon = sigma * sqrt(M + 2*sqrt(M));

%im = imread('../validationset/153_1.png');
%x = im2double(im);
%x = NormalizeImageResize('../validationset/153_1.png');
x = addNoiseResize('C:\Users\lewis\Desktop\CW1/validationset/153_1.png');
y = add_noise(Phit(x));
load m3trainednet7mb10epochs;
max_iter = 100;
imshow(x);

imshow(cell2mat(compute_net(net, x, 64)));

% dn_x is the resulting denoised image
%dn_x = PnP(x, y, net, delta, Phi, Phit, rel_tol, rel_tol2, epsilon, max_iter);
%imshow(dn_x);

%% 7. M3 validation ...


% takes an image x, measurements y, a DnCNN, a step size,
% a Phi and a Phit,
% tolerances for loss function image compared to prev iter,
% as well as an epsilon and a max number of iterations
% returns a denoised image
function dn_x = PnP(x, y, net, delta, Phi, Phit, rel_tol, rel_tol2, epsilon, max_iter)
    i = 1;
    s = -y;
    n = s*min(epsilon/norm(s(:)), 1);
    v = zeros(size(y));
    
    xprev = zeros(size(Phi(s)));
    
    while(((norm(y - Phit(x), 2) <= epsilon) || (norm(x(:) - xprev(:), 2)/norm(x(:), 2) <= rel_tol2) || (i < max_iter)))
        xprev = x;
        difference = x - delta*real(Phi(s + n - v));
        x(x > 1) = 1;
        x(x < 0) = 0;
        x = cell2mat(compute_net(net, difference, 64));
        s = Phit(x) - y;
        z = v - s;
        n = z*min(epsilon/norm(z(:)), 1);
        v = v - (s + n);
        
        i = i + 1;
        fprintf("Iterations: %d\n", i);
        imshow(x);
    end
    dn_x = x;
end


%% 2. Normalization to the training and testing datasets
% Working with a normalized data set can help enhance the training of
% networks. This can be achieved with the following function.
function img_res = NormalizeImageResize(file)
%   This function takes as input:
%       - file: an image directory.
%   It returns:
%       - img_res: the image in range [0,1] with dimensions [256x256].
    img = imread(file);
    img_res = im2double(img);

end

function out = addNoiseResize(file)
    % Noise addition operator
    sigma = 0.07;
    add_noise_to_picture = @(x) x + sigma*randn(size(x));
    out = add_noise_to_picture(NormalizeImageResize(file));
end

%% 4. Working with image patches
% The training of a denoiser is eased when using large number of images
% (batch size) and small image size (patches). We suggest that you train
% your network on a large number (>>1000) of patches of size 64x64, created
% from the original 1000 groundtruth images of size 256x256 in the training
% data set. The network will then act as a denoiser on 64x64 images.
% Your problem is to validate the method on the 256x256 images of the
% testing data set. The following function will allow you to
% apply your network to images of any size, by simply denoising 64x64 patches
% separately with the network.
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
