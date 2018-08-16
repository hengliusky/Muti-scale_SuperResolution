% function [] = augment(director, pattern)

% d = fullfile(director, pattern);
d = fullfile('./Train/', '*.bmp');
files = dir(d);

result = cell(numel(files), 1);

for i = 1 : numel(result)
%     result{i} = fullfile(director, files(i).name);
    result{i} = fullfile('./Train/', files(i).name);
    img = imread(result{i});
    
    im1 = imresize(img, 0.9);
    im2 = imresize(img, 0.8);
    im3 = imresize(img, 0.7);
    im4 = imresize(img, 0.6);
    
    name = (files(i).name(1:(length(files(i).name)-4)));
    
    imwrite(im1, [name, '_01', '.bmp']);
    imwrite(im2, [name, '_02', '.bmp']);
    imwrite(im3, [name, '_03', '.bmp']);
    imwrite(im4, [name, '_04', '.bmp']);
    %% Ðý×ªÍ¼Ïñ
%     im1 = imrotate(img, 0);
%     im2 = imrotate(img, 90);
%     im3 = imrotate(img, 180);
%     im4 = imrotate(img, 270);
% 
%     %% ·­×ªÍ¼Ïñ
%     im_h1 = flip(im1,1);
%     im_v1 = flip(im1,2);
% 
%     im_h2 = flip(im2,1);
%     im_v2 = flip(im2,2);
% 
%     im_h3 = flip(im3,1);
%     im_v3 = flip(im3,2);
% 
%     im_h4 = flip(im4,1);
%     im_v4 = flip(im4,2);
%     
%     name = (files(i).name(1:(length(files(i).name)-4)));
%     
% %     imwrite(im1, [name, '_1', '.bmp']);
%     imwrite(im2, [name, '_2', '.bmp']);
%     imwrite(im3, [name, '_3', '.bmp']);
%     imwrite(im4, [name, '_4', '.bmp']);
%     imwrite(im_h1, [name, '_5', '.bmp']);
%     imwrite(im_h2, [name, '_6', '.bmp']);
%     imwrite(im_h3, [name, '_7', '.bmp']);
%     imwrite(im_h4, [name, '_8', '.bmp']);
%     imwrite(im_v1, [name, '_9', '.bmp']);
%     imwrite(im_v2, [name, '_10', '.bmp']);
%     imwrite(im_v3, [name, '_11', '.bmp']);
%     imwrite(im_v4, [name, '_12', '.bmp']);
end
