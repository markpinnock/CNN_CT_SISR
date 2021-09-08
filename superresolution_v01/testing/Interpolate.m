% HI_PATH = "Z:/SISR_Data/Toshiba/Mat_Real_Hi/";
LO_PATH = "Z:/SISR_Data/Toshiba/Mat_Lo/";
SAVE_PATH = "D:/SISR_Data/Toshiba/Mat_Real_Int/";

% hi_list = dir(HI_PATH);
% hi_list = hi_list(3:end);
lo_list = dir(LO_PATH);
lo_list = lo_list(3:end);

x = 1:512;
y = 1:512;
z = [3, 7, 11];

[X Y Z] = ndgrid(1:512, 1:512, 1:12);
% MSE = zeros(1, 35);
% pSNR = zeros(1, 35);
% SSIM = zeros(1, 35);

for i = 1:35
%     hi = load(strcat(HI_PATH, hi_list(i).name)).hi;
    lo = load(strcat(LO_PATH, lo_list(i).name)).lo;
    lo = imgaussfilt3(lo, [1, 1, 1]);
    int = interpn(x, y, z, lo, X, Y, Z, 'spline');
    save(strcat(SAVE_PATH, lo_list(i).name), 'int');

%     MSE(i) = mean((hi - int).^2, 'all');
%     pSNR(i) = psnr(int, hi);
%     SSIM(i) = ssim(int, hi);
    fprintf("%d\n", i);
end