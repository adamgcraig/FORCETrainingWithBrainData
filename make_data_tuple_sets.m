%% Make data tuple sets
% Adam Craig @ HKBU, 2023-01-19.
% Our main goal:
% Divide the HCP data into training, validation, and testing sets.
% We have a list of 838 subjects from whom we have
% fMRI-derived BOLD response time series data,
% DT-MRI-derived white matter structural connectivity (SC) data, and
% T1- and/or T2-weighted structural MRI-derived anatomical features.
% Load the list of subjects, and
% check that we actually have all the data for each subject.
% Each should have 4 fMRI time series:
% left-to-right and right-to-left for each of 2 separate days.
% Check that each has all 1200 time points for all 360 brain areas.
% The SC data should be symmetric, non-negative 360x360 matrices.
% The 4 anatomical measures are
% thickness, myelination, curvature, and sulcus depth.
% We should have four values per brain area, per subject.
% The first thing we want to do with the data is
% train a machine learning model to predict the BOLD signal of a brain area
% from its previous BOLD signal and those of its neighbors.
% Ideally, the model would capture some universal rule of brain dynamics,
% possibly parameterized by the anatomical features.
% We may instead need to train separate models for different brain areas.
% We may then want to further refine the training on a single individual.
% We want to train any individual model on the training data,
% use the validation data to compare different versions of the model
% with different hyperparameters,
% and use the testing data to test the one with the best validation score.
% These considerations suggest two ways of splitting the data.
% 1. Place each user in the training, validation, or testing set,
% keeping all their fMRI runs in the same set.
% 2. Split each user, placing 2 runs in the training set,
% 1 in the validation set, and 1 in the testing set.

%% Load the subject list, and save it as a text file and a binary.
S = load('C:\Users\agcraig\Documents\DataFromDaniel\ListSubj_838.mat');
subject_ids = S.Subject;
num_subjects = numel(subject_ids);
writematrix(subject_ids,'C:\\Users\\agcraig\\Documents\\HCP_data\\ListSubj_838.txt');
list_fid = fopen('C:\\Users\\agcraig\\Documents\\HCP_data\\ListSubj_838.bin','w');
fwrite(list_fid,subject_ids,'uint32');
fclose(list_fid);

%% Check that every subject has complete fMRI data.
disp('Check that every subject has complete fMRI data...')
fmri_suffixes = {'1_LR' '1_RL' '2_LR' '2_RL'};
num_fmri_suffixes = numel(fmri_suffixes);
fmri_file_size = zeros(num_subjects,num_fmri_suffixes);
for subject_index = 1:num_subjects
    subject_id = subject_ids(subject_index);
    for suffix_index = 1:num_fmri_suffixes
        fmri_suffix = fmri_suffixes{suffix_index};
        fmri_file = sprintf('C:\\Users\\agcraig\\Documents\\HCP_data\\fMRI_binaries\\ts_%u_%s.bin',subject_id,fmri_suffix);
        if exist(fmri_file,'file') ~= 0
            file_info = dir(fmri_file);
            fmri_file_size(subject_index,suffix_index) = file_info.bytes;
        end
    end
end
% Each should have 1200 time points for 360 areas, stored as 8-byte floats.

fmri_file_exists = fmri_file_size > 0;
has_all_fmri = all(fmri_file_exists,2);
num_with_all_fmri = nnz(has_all_fmri);
fprintf('%u subjects have all their fMRI files.\n',num_with_all_fmri)

num_time_points = 1200;
num_brain_areas = 360;
expected_bytes = 1200 * num_brain_areas * 8;
fmri_file_incomplete = fmri_file_size ~= expected_bytes;
num_incomplete_fmris = nnz(fmri_file_incomplete);
fprintf('%u fMRI files are not the expected size.\n',num_incomplete_fmris)

%% Export the fMRI data to binary files split up by subject and brain area.
for subject_index = 1:num_subjects
    subject_id = subject_ids(subject_index);
    for suffix_index = 1:num_fmri_suffixes
        fmri_suffix = fmri_suffixes{suffix_index};
        fmri_in_file = sprintf('C:\\Users\\agcraig\\Documents\\HCP_data\\fMRI_binaries\\ts_%u_%s.bin',subject_id,fmri_suffix);
        fmri_in_id = fopen(fmri_in_file,'r');
        fmri_data = reshape( fread(fmri_in_id,num_brain_areas*num_time_points,'float64'), num_brain_areas, num_time_points );
        fclose(fmri_in_id);
        for brain_area_index = 1:num_brain_areas
            fmri_out_file = sprintf('C:\\Users\\agcraig\\Documents\\HCP_data\\fMRI_binaries_1_area\\ts_%u_%s_%u.bin',subject_id,fmri_suffix,brain_area_index);
            fmri_out_id = fopen(fmri_out_file,'w');
            fwrite( fmri_out_id, fmri_data(brain_area_index,:), 'float64' );
            fclose(fmri_out_id);
        end
    end
end

%% Compare FC from Daniel's data to FC computed from Wang Rong's data.
% S = load('C:\Users\agcraig\Documents\DataFromDaniel\FC.mat');
% fc_data_all = S.FC;
% fc_data_first = permute( fc_data_all(1,:,:), [2 3 1] );
% subject_id = subject_ids(1);
% fmri_cell = cell(4,1);
% for suffix_index = 1:num_fmri_suffixes
%     fmri_suffix = fmri_suffixes{suffix_index};
%     fmri_file = sprintf('C:\\Users\\agcraig\\Documents\\HCP_data\\fMRI_binaries\\ts_%u_%s.bin',subject_id,fmri_suffix);
%     fid = fopen(fmri_file);
%     fmri_cell{suffix_index} = reshape( fread(fid,360*1200,'float64'), 360, 1200 );
%     fclose(fid);
% end
% fmri_ts_all = horzcat(fmri_cell{:});
% fc_data_from_ts = corr(fmri_ts_all');
% fc_data_diff = fc_data_from_ts - fc_data_first;
% figure
% boxplot( fc_data_diff(:) )

%% What happens when we swap the hemispheres in Daniel's FC matrix?
% They supposedly go from right to left
% instead of left to right, as in Wang Rong's data.

% fc_data_first_rr = fc_data_first(1:180,1:180);
% fc_data_first_rl = fc_data_first(1:180,181:360);
% fc_data_first_lr = fc_data_first(181:360,1:180);
% fc_data_first_ll = fc_data_first(181:360,181:360);
% fc_data_first_swapped = [
%     fc_data_first_ll fc_data_first_lr;
%     fc_data_first_rl fc_data_first_rr
%     ];
% fc_data_diff_swapped = fc_data_from_ts - fc_data_first_swapped;
% figure
% boxplot( fc_data_diff_swapped(:) )
% figure
% scatter( fc_data_from_ts(:), fc_data_first_swapped(:) )
% title('L-R-swapped FC from Daniel vs FC made from time series from Wang Rong')
% xlabel('FC from time series from Wang Rong')
% ylabel('FC from Daniel with left and right hemispheres swapped')

%% Convert the structural connectivities to ranks.
S = load('C:\Users\agcraig\Documents\DataFromDaniel\SC.mat');
sc_data_all = S.SC_RL;
% sc_data_first = sc_data_all(:,:,1);

% Compare to SC from the other HCP dataset we got from Wang Rong.
% fid = fopen('C:\Users\agcraig\Documents\HCP_data\dtMRI_binaries\sc_100206.bin','r');
% sc_data_from_binary = reshape( fread(fid,360*360,'float64'), 360, 360 );
% fclose(fid);
% figure
% scatter( sc_data_first(:), sc_data_from_binary(:) )
% title('subject 100206')
% xlabel('SC values from Daniel')
% ylabel('SC values from Wang Rong')

% Compare to SC from the other HCP dataset we got from Wang Rong
% after swapping the left and right hemisphere order.
% Wang Rong's version has the left hemisphere first, then the right.
% Daniel's has the right hemisphere first, then the left.
% fid = fopen('C:\Users\agcraig\Documents\HCP_data\dtMRI_binaries\sc_100206.bin','r');
% sc_data_from_binary = reshape( fread(fid,360*360,'float64'), 360, 360 );
% fclose(fid);
% sc_data_first_rr = sc_data_first(1:180,1:180);
% sc_data_first_rl = sc_data_first(1:180,181:360);
% sc_data_first_lr = sc_data_first(181:360,1:180);
% sc_data_first_ll = sc_data_first(181:360,181:360);
% sc_data_first_swapped = [
%     sc_data_first_ll sc_data_first_lr;
%     sc_data_first_rl sc_data_first_rr
%     ];
% figure
% scatter( sc_data_first_swapped(:), sc_data_from_binary(:) )
% title('subject 100206 swapped so that left hemisphere comes first')
% xlabel('SC values from Daniel')
% ylabel('SC values from Wang Rong')
% figure
% sc_diff = sc_data_first_swapped - sc_data_from_binary;
% boxplot( sc_diff(:) )
% disp(   max(  abs( sc_diff(:) )  )   )

sc_data_all_rr = sc_data_all(1:180,1:180,:);
sc_data_all_rl = sc_data_all(1:180,181:360,:);
sc_data_all_lr = sc_data_all(181:360,1:180,:);
sc_data_all_ll = sc_data_all(181:360,181:360,:);
sc_data_all = [
    sc_data_all_ll sc_data_all_lr;
    sc_data_all_rl sc_data_all_rr
    ];
num_brain_areas = size(sc_data_all,1);
[~, brain_area_sc_rankings] = sort( sc_data_all, 1, 'descend' );
for subject_index = 1:num_subjects
    subject_id = subject_ids(subject_index);
    for brain_area_index = 1:num_brain_areas
        sc_rank_file = sprintf('C:\\Users\\agcraig\\Documents\\HCP_data\\sc_rank_binaries\\sc_rank_%u_%u.bin',subject_id,brain_area_index);
        scr_file_id = fopen(sc_rank_file,'w');
        fwrite( scr_file_id, brain_area_sc_rankings(:,brain_area_index,subject_index), 'uint32' );
        fclose(scr_file_id);
    end
end

%% Convert the anatomical data into individual binary files.

S = load('C:\Users\agcraig\Documents\DataFromDaniel\Structural_measures.mat');
anatomy = permute( S.properties, [3 1 2] );
% Swap the order here so that the left hemisphere is first, then the right.
anatomy = anatomy(:,:,[181:360 1:180]);
for subject_index = 1:num_subjects
    subject_id = subject_ids(subject_index);
    for brain_area_index = 1:num_brain_areas
        anat_file = sprintf('C:\\Users\\agcraig\\Documents\\HCP_data\\anatomy_binaries\\anatomy_%u_%u.bin',subject_id,brain_area_index);
        anat_file_id = fopen(anat_file,'w');
        fwrite( anat_file_id, anatomy(:,subject_index,brain_area_index), 'float64' );
        fclose(anat_file_id);
    end
end
% anat_file_id = fopen(anat_file,'r');
% anat_read_back_test = fread( anat_file_id, 4, 'float64' );
% fclose(anat_file_id);
% disp( anat_read_back_test - anatomy(:,subject_index,brain_area_index) )
