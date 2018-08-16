%% ���ļ����ڲ���degree�ؽ�Ч��
close all;
clear all;

%% 多尺度网络，放大因子为4
addpath('./matlab');

%% ģ��·���� �������ã� ѵ������

model_dir = './model/rect2/';
net_model = ['./test_rect1.prototxt'];
net_weights = [model_dir 'mutiscale_rect2_iter_205000.caffemodel']; %160000 223500
phase = 'test';


%% ����gpu��ָ��id
% caffe.set_mode_gpu();
% caffe.set_device(1);
caffe.set_mode_cpu();
%% �������粢����Ȩֵ
net = caffe.Net(net_model, net_weights, phase);

%% ����ͼ��
% img = imread('../Test/Set5/butterfly_GT.bmp');  %256*256
% img = imread('../Test/Set5/baby_GT.bmp');   %512*512
% img = imread('../Test/Set5/woman_GT.bmp');  %228*344
% img = imread('../Test/Set5/bird_GT.bmp');  %288*288
% img = imread('../Test/Set5/head_GT.bmp');   %280*280

% img = imread('../Test/Set14/monarch.bmp');  %768*512
% img = imread('../Test/Set14/baboon.bmp');  %500*480
% img = imread('../Test/Set14/lenna.bmp');  %512*512
% img = imread('../Test/Set14/ppt3.bmp');  %528*656
img = imread('../Test/Set14/foreman.bmp');  %352*288
% img = imread('../Test/Set14/barbara.bmp');  %720*576
% img = imread('../Test/Set14/zebra.bmp');  %584*388
% img = imread('../Test/Set14/flowers.bmp');  %500*360
% img = imread('../Test/Set14/face.bmp');  %276*276
% img = imread('../Test/Set14/pepper.bmp');  %512*512
% img = imread('../Test/Set14/comic.bmp');  %248*360
% img = imread('../Test/Set14/bridge.bmp');  %512*512
% img = imread('../Test/Set14/coastguard.bmp');  %352*288
% img = imread('../Test/Set14/man.bmp');  %512*512

up_scale = 4;

%% 原始图像为彩色图像
if size(img, 3) > 1
im = rgb2ycbcr(img);    %ת��ɫ�ʿռ�
im_l_y = im(:,:,1);
im_l_cb = im(:,:,2);
im_l_cr = im(:,:,3);


%% �ü�ground truth�����ڼ����ɫͼ�������
img = modcrop(img, up_scale);

im_gnd = modcrop(im_l_y, up_scale);
im_gnd = single(im_gnd)/255;

[hei, wid] = size(im_gnd);
im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
im_b = imresize(im_l, [hei, wid], 'bicubic');

im_l_cb = modcrop(im_l_cb, up_scale);
im_l_cb = single(im_l_cb)/255;
im_l_cr = modcrop(im_l_cr, up_scale);
im_l_cr = single(im_l_cr)/255;

%% extract edge
[ledge, or, ft] = phasecong3(im_b,3,6,3,1.5,0.10,2.0,0.1,10);
[hedge, hr_or, hr_ft] = phasecong3(im_gnd,3,6,3,1.5,0.10,2.0,0.1,10);
clear or;
clear ft;

%% SRCNN �ؽ�
x4 = 'model/9-5-5(ImageNet)/x4.mat';
% im_h_cnn = SRCNN(x3, im_b);
im_h_cnn = SRCNN(x4, im_b);

%% mutiscale �ؽ�
input_data = {im_b, ledge};
output = net.forward(input_data);   

% [nrow, ncol] = size(im_h_cnn);
% im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
% im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');

%% remove border 
% im_h_mutiscale = shave(uint8(output{1} * 255), [up_scale, up_scale]);
% % im_h_cnn = shave(uint8(im_h_cnn * 255), [up_scale, up_scale]);
% im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
% im_b = shave(uint8(im_b * 255), [up_scale, up_scale]);

im_h_mutiscale = shave(uint8(output{2} * 255), [up_scale, up_scale]);
im_h_cnn = shave(uint8(im_h_cnn * 255), [up_scale, up_scale]);
im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
im_b = shave(uint8(im_b * 255), [up_scale, up_scale]);
% imwrite(im_h_mutiscale,'./test_adam/img_adam_muti/man.bmp')
% imwrite(im_h_cnn,'./test_sgd/single_loss/srcnn_img/man.bmp')
% imwrite(im_b,'./single_loss/bic/butterfly.bmp')
% imwrite(im_gnd,'./single_loss/gnd/butterfly.bmp')

%% computer PSNR
psnr_bic = compute_psnr(im_gnd, im_b);
psnr_srcnn = compute_psnr(im_gnd, im_h_cnn);
psnr_mutiscale = compute_psnr(im_gnd, im_h_mutiscale);

%% computer SSIM
ssim_bic  = ssim_index(im_gnd, im_b);%, [0.01 0.03], ones(8));
ssim_srcnn  = ssim_index(im_gnd, im_h_cnn);%, [0.01 0.03], ones(8));
ssim_mutiscale  = ssim_index(im_gnd, im_h_mutiscale);%, [0.01 0.03], ones(8));

fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bic);
fprintf('PSNR for SRCNN Reconstruction: %f dB\n', psnr_srcnn);
fprintf('PSNR for mutiscale Reconstruction: %f dB\n', psnr_mutiscale);
fprintf('---------------------------------------\n');
fprintf('---------------------------------------\n');
fprintf('SSIM for Bicubic Interpolation: %f dB\n', ssim_bic);
fprintf('SSIM for SRCNN Reconstruction: %f dB\n', ssim_srcnn);
fprintf('SSIM for mutiscale Reconstruction: %f dB\n', ssim_mutiscale);

figure, imshow(im_b); title('Bicubic Interpolation');
figure, imshow(im_h_cnn); title('SRCNN Reconstruction');
figure, imshow(im_h_mutiscale); title('mutiscale Reconstruction');
figure, imshow(im_gnd); title('Ground Truth');

end

%% 原始图像为灰度图像
% if size(img, 3) == 1
%    % im = rgb2ycbcr(img);    %ת��ɫ�ʿռ�
%    % im_l_y = im(:,:,1);
%    % im_l_cb = im(:,:,2);
%    % im_l_cr = im(:,:,3);
%     im_l_y = img;
% 
% %% ground truth
% img = modcrop(img, up_scale);
% 
% im_gnd = modcrop(im_l_y, up_scale);
% 
% im_gnd = single(im_gnd)/255;
% 
% %% bicubic interpolation
% [hei, wid] = size(im_gnd);
% im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
% im_b = imresize(im_l, [hei, wid], 'bicubic');
% 
% %% SRCNN， 需要换成放大因子为4的
% % x3 = 'model/9-1-5(91 images)/x4.mat';   
% x3 = 'model/9-5-5(ImageNet)/x4.mat';
% im_h_cnn = SRCNN(x3, im_b);
% 
% %% extract edge
% [ledge, or, ft] = phasecong3(im_b,3,6,3,1.5,0.10,2.0,0.1,10);
% 
% %% mutiscale 
% input_data = {im_b, ledge};
% output = net.forward(input_data);   
% % conv7_fea = net.blobs('conv7').get_data();
% % 
% % input_label = {im_gnd};
% % net.blobs('conv8').set_data(input_label);
% 
% % im_h_dsn = uint8(output{1});
% % im_h_dsn = uint8(output{1}*255);
% % im_b = uint8(im_b*255);
% 
% %% remove border 
% im_h_mutiscale = shave(uint8(output{1} * 255), [up_scale, up_scale]);
% im_h_cnn = shave(uint8(im_h_cnn * 255), [up_scale, up_scale]);
% im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
% im_b = shave(uint8(im_b * 255), [up_scale, up_scale]);
% 
% 
% %% computer PSNR
% psnr_bic = compute_psnr(im_gnd, im_b);
% psnr_srcnn = compute_psnr(im_gnd, im_h_cnn);
% psnr_mutiscale = compute_psnr(im_gnd, im_h_mutiscale);
% 
% %% computer SSIM
% ssim_bic  = ssim_index(im_gnd, im_b);%, [0.01 0.03], ones(8));
% ssim_srcnn  = ssim_index(im_gnd, im_h_cnn);%, [0.01 0.03], ones(8));
% ssim_mutiscale  = ssim_index(im_gnd, im_h_mutiscale);%, [0.01 0.03], ones(8));
% 
% fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bic);
% fprintf('PSNR for SRCNN Reconstruction: %f dB\n', psnr_srcnn);
% fprintf('PSNR for mutiscale Reconstruction: %f dB\n', psnr_mutiscale);
% fprintf('---------------------------------------\n');
% fprintf('---------------------------------------\n');
% fprintf('SSIM for Bicubic Interpolation: %f dB\n', ssim_bic);
% fprintf('SSIM for SRCNN Reconstruction: %f dB\n', ssim_srcnn);
% fprintf('SSIM for mutiscale Reconstruction: %f dB\n', ssim_mutiscale);
% 
% figure, imshow(im_b); title('Bicubic Interpolation');
% figure, imshow(im_h_cnn); title('SRCNN Reconstruction');
% figure, imshow(im_h_mutiscale); title('mutiscale Reconstruction');
% 
% figure, imshow(im_gnd); title('Ground Truth');
% 
% end

caffe.reset_all();

[nrow, ncol] = size(im_h_cnn);
% im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
% im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');
% 
% % %% im_h_bicΪbicubic��ֵͼ��
% im_h_bic = zeros([nrow, ncol, 3]);
% im_h_bic(:, :, 1) = single(im_b)/255;
% im_h_bic(:, :, 2) = im_h_cb;
% im_h_bic(:, :, 3) = im_h_cr;
% im_bic = ycbcr2rgb(im_h_bic);
% 
% % %% im_hΪSRCNN���ؽ�ͼ��
% im_h_ycbcr = zeros([nrow, ncol, 3]);
% im_h_ycbcr(:, :, 1) = single(im_h_cnn)/255;
% im_h_ycbcr(:, :, 2) = im_h_cb;
% im_h_ycbcr(:, :, 3) = im_h_cr;
% im_h_srcnn = ycbcr2rgb(im_h_ycbcr);
% 
% % %% im_h_depnet Ϊmutiscale���ؽ�ͼ��
% im_h_mutiscale_ycbcr = zeros([nrow, ncol, 3]);
% im_h_mutiscale_ycbcr(:, :, 1) = single(im_h_mutiscale)/255;
% im_h_mutiscale_ycbcr(:, :, 2) = im_h_cb;
% im_h_mutiscale_ycbcr(:, :, 3) = im_h_cr;
% im_h_mutiscale = ycbcr2rgb(im_h_mutiscale_ycbcr);
% 
% gnd = zeros([nrow, ncol, 3]);
% gnd(:, :, 1) = single(im_gnd)/255;
% gnd(:, :, 2) = im_h_cb;
% gnd(:, :, 3) = im_h_cr;
% gnd = ycbcr2rgb(gnd);
% 
% imwrite(im_h_mutiscale,'./test_sgd/color/muti/foreman.bmp')
% imwrite(im_h_srcnn,'./test_sgd/color/srcnn/foreman.bmp')
% imwrite(im_bic,'./test_sgd/color/bic/foreman.bmp')
% imwrite(gnd,'./test_sgd/color/gnd/foreman.bmp')
% 
% % %% ��ʾ�ؽ����
% figure, imshow(im_bic); title('Bicubic Interpolation');
% figure, imshow(im_h_srcnn); title('SRCNN Reconstruction');
% figure, imshow(im_h_mutiscale); title('mutiscale Reconstruction');
% figure, imshow(gnd); title('Ground Truth');

%% ���������
% im_gnd = uint8(im_gnd*255);
% im_b = uint8(im_b*255);
% im_h_cnn = uint8(im_h_cnn*255);
% output{1} = uint8(output{1}*255);

% psnr_bic = compute_psnr(im_gnd, im_b);
% psnr_srcnn = compute_psnr(im_gnd, im_h_cnn);
% psnr_mutiscale = compute_psnr(im_gnd, output{1});

% %% ����SSIM
% ssim_bic  = ssim_index(im_gnd, im_b);%, [0.01 0.03], ones(8));
% ssim_srcnn  = ssim_index(im_gnd, im_h_cnn);%, [0.01 0.03], ones(8));
% ssim_mutiscale  = ssim_index(im_gnd, output{1});%, [0.01 0.03], ones(8));

%% ��������
% fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bic);
% fprintf('PSNR for SRCNN Reconstruction: %f dB\n', psnr_srcnn);
% fprintf('PSNR for mutiscale Reconstruction: %f dB\n', psnr_mutiscale);
% fprintf('---------------------------------------\n');
% fprintf('---------------------------------------\n');
% fprintf('SSIM for Bicubic Interpolation: %f dB\n', ssim_bic);
% fprintf('SSIM for SRCNN Reconstruction: %f dB\n', ssim_srcnn);
% fprintf('SSIM for mutiscale Reconstruction: %f dB\n', ssim_mutiscale);
% end

% caffe.reset_all();
% imwrite(im_bic, ['Bicubic Interpolation' '.bmp']);
% imwrite(im_h_srcnn, ['SRCNN Reconstruction' '.bmp']);
% imwrite(im_h_mutiscale, ['mutiscale Reconstruction' '.bmp']);
% imwrite(img, ['Ground Truth Reconstruction' '.bmp']);
