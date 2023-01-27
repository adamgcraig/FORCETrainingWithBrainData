function [target_area_ts,neighbor_ts,area_features,subject_id,time_series_name,brain_area_index] = randomly_select_fmri_data(data_dir,subject_ids,time_series,num_brain_areas,num_time_steps,num_area_features,num_neighbor_areas,use_mean_sc)
%RANDOMLY_SELECT_INPUT_DATA Randomly select fMRI time series data.
%   1. Randomly select a subject ID from the list.
%   2. Randomly select a time series from that subject.
%   3. Randomly select a target brain area from that time series.
%   4. Retrieve the anatomical feature data for that brain area.
%   5. Retrieve the fMRI data for that brain area and time series.
%   6. Select the num_neighbors brain areas
%      that are most strongly connected to this area.
%   7. Load the fMRI data for those areas from the same time series.

area_feature_dir = 'anatomy_binaries';
fmri_dir = 'fMRI_binaries_1_area';
sc_rank_dir = 'sc_rank_binaries';

%   1. Randomly select a subject ID from the list.
subject_id = subject_ids(  randi( numel(subject_ids) )  );
%   2. Randomly select a time series from that subject.
time_series_name = time_series{randi( numel(time_series) )};
%   3. Randomly select a target brain area from that time series.
brain_area_index = randi(num_brain_areas);
%   4. Retrieve the anatomical feature data for that brain area.
af_file = fopen( [data_dir filesep area_feature_dir filesep sprintf('anatomy_%u_%u.bin',subject_id,brain_area_index)],'r');
area_features = fread(af_file,num_area_features,'float64');
fclose(af_file);
%   5. Retrieve the fMRI data for that brain area and time series.
fmri_file = fopen( [data_dir filesep fmri_dir filesep sprintf('ts_%u_%s_%u.bin',subject_id,time_series_name,brain_area_index)],'r');
target_area_ts = fread(fmri_file,num_time_steps,'float64')';
fclose(fmri_file);
%   6. Select the num_neighbors brain areas
%      that are most strongly connected to this area.
if use_mean_sc
    sc_subject = 'mean';
else
    sc_subject = num2str(subject_id);
end
sc_rank_file = fopen( [data_dir filesep sc_rank_dir filesep sprintf('sc_rank_%s_%u.bin',sc_subject,brain_area_index)],'r');
sc_ranks = fread(sc_rank_file,num_neighbor_areas,'uint32');
fclose(sc_rank_file);
%   7. Load the fMRI data for those areas from the same time series.
neighbor_ts = NaN(num_neighbor_areas,num_time_steps);
for neighbor_index = 1:num_neighbor_areas
    neighbor_brain_area = sc_ranks(neighbor_index);
    neighbor_fmri_file = fopen( [data_dir filesep fmri_dir filesep sprintf('ts_%u_%s_%u.bin',subject_id,time_series_name,neighbor_brain_area)],'r');
    neighbor_ts(neighbor_index,:) = fread(fmri_file,num_time_steps,'float64')';
    fclose(neighbor_fmri_file);
end

end