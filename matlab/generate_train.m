clear;close all;
%% settings
%% 188352
folder = '../Train aug';
savepath = 'train.h5';
size_input = 50;
size_label = 50;
size_lredge = 50;
size_hredge = 50;
scale = 4;
stride = 14;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
ledge = zeros(size_lredge, size_lredge, 1, 1);
hedge = zeros(size_hredge, size_hredge, 1, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
    
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
    
    im_label = modcrop(image, scale);
    [hei,wid] = size(im_label);
    im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');
    
     %% ������λһ������ȡ��Ե
    [lr_edge, lr_or, lr_ft] = phasecong3(im_input,3,6,3,1.5,0.10,2.0,0.1,10);
    [hr_edge, hr_or, hr_ft] = phasecong3(im_label,3,6,3,1.5,0.10,2.0,0.1,10);

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
            sublr_edge = lr_edge(x : x+size_lredge-1, y : y+size_lredge-1);
            subhr_edge = hr_edge(x : x+size_hredge-1, y : y+size_hredge-1);

            count=count+1;
            data(:, :, 1, count) = subim_input;
            label(:, :, 1, count) = subim_label;
            ledge(:, :, 1, count) = sublr_edge;
            hedge(:, :, 1, count) = subhr_edge;
        end
    end
end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 
ledge = ledge(:, :, 1, order);
hedge = hedge(:, :, 1, order);

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);
    batchledge = ledge(:,:,1,last_read+1:last_read+chunksz); 
    batchhedge = hedge(:,:,1,last_read+1:last_read+chunksz); 

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1], 'led', [1,1,1,totalct+1], 'hed', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, batchledge, batchhedge, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
