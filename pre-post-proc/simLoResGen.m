function simLoResGen(file_path, save_path)
    subject_list = dir(file_path);
    subject_list = subject_list(3:end);
    
    if ~exist(save_path, 'dir')
       mkdir(save_path);
    end
    
    save_path = strcat(save_path, '\');

    TOSHIBA_RANGE = [-2917, 16297];
    min_val = TOSHIBA_RANGE(1);
    max_val = TOSHIBA_RANGE(2);
    lo_res = [512, 512, 3];
    scaling = 50;
    poiss_ns = 5e5;
    gauss_mn = 0;
    gauss_s = 1;

    for i = 4:4%length(subject_list)
        curr_path = strcat(file_path, '\', subject_list(i).name);
        subject_name = char(subject_list(i).name);
        
        if subject_name(end-1:end) == '_2'
            study_num = 2;
            subject_name = string(subject_name(1:end-2));
        else
            study_num = 1;
            subject_name = string(subject_name);
        end

        file_list = dir(curr_path);
        file_list = file_list(3:end);

        for j = 1:length(file_list)
            file_name = file_list(j).name;
            img_path = strcat(curr_path, '\', file_name);
            series_str = "";
            
            for k = 1:5
                if strcmp(file_name(k), " ")
                    break;
                end
                
                series_str = series_str + file_name(k);
            end
            
            series_num = str2num(series_str);
            
            vol = double(niftiread(strcat(img_path)));
            vol = (vol - min_val) / (max_val - min_val);

            vol_size = size(vol);
            num_vols = floor(vol_size(3) / 12);
            counter = 1;
            
            for l = 1:num_vols
                hvol = vol(:, :, (l * 12) - 11:l * 12);
                lvol = zeros(lo_res);
                
                for m = 1:3
                    lo_img = mean(imgaussfilt3(hvol(:, :, (m * 4) - 3:m * 4), [0.01 0.01 1]), 3);
                    [rslice, ~] = radon(lo_img, 0:179);
                    rslice = rslice / scaling;
                    trans = poissrnd(exp(-rslice) * poiss_ns) + normrnd(zeros(size(rslice)) + gauss_mn, ones(size(rslice)) * gauss_s);
                    proj = log(poiss_ns ./ trans);
                    noise_img = iradon(proj * scaling, 0:179, 512);
                    lvol(:, :, m) = noise_img;
                end
                
                hi_save_name = sprintf("%s_%d_%03d_%03d_H.mat", subject_name, study_num, series_num, counter);
                lo_save_name = sprintf("%s_%d_%03d_%03d_L.mat", subject_name, study_num, series_num, counter);
                save(strcat(save_path, hi_save_name), 'hvol');
                save(strcat(save_path, lo_save_name), 'lvol');
                fprintf("%s, %s SAVED\n", hi_save_name, lo_save_name);
                counter = counter + 1;
            end
        end
    end
end