function file_list = make_fmri_file_list(subjects,time_series,num_brain_areas)
%MAKE_FMRI_FILE_LIST Make a list of single-area fMRI time series files.
%   These are the binary files with file names in format
%   [subject ID]_[time series string]_[brain area index].bin

if ~exist('do_shuffle','var')
    do_shuffle = true;
end
num_subjects = numel(subjects);
num_time_series_per_subject = numel(time_series);
brain_areas = num2cell(1:num_brain_areas);
[SUBJECT_index, TIME_SERIES_index, BRAIN_AREA_index] = meshgrid(1:num_subjects,1:num_time_series_per_subject,1:num_brain_areas);
SUBJECT = subjects(SUBJECT_index);
TIME_SERIES = time_series(TIME_SERIES_index);
BRAIN_AREA = brain_areas(BRAIN_AREA_index);
file_list = cellfun( @(s,ts,ba) sprintf('%s_%s_%u.bin',s,ts,ba), SUBJECT, TIME_SERIES, BRAIN_AREA, 'UniformOutput', false );

end