function f = subplot_time_series(time_series,num_rows,times,time_units,channel_units,same_scale)
%SUBPLOT_TIME_SERIES Plot each figure in its own channel.
%   time_series: a multi-channel time series with channels in rows,
%                time points in columns
%   num_rows: number of rows in the subplot,
%             defaults to the number of channels

f = figure('Position',[0 0 1000 1000]);
num_channels = size(time_series,1);
if ~exist('num_rows','var')
    num_rows = num_channels;
end
if ~exist('times','var')
    times = 1:size(time_series,2);
end
if ~exist('time_units','var')
    time_units = 'steps';
end
if ~exist('channel_units','var')
    channel_units = '';
end
if ~exist('same_scale','var')
    same_scale = true;
end
if same_scale
    y_bounds = [ min(time_series,[],'all') max(time_series,[],'all') ];
end
num_cols = ceil(num_channels/num_rows);
for channel_index = 1:num_channels
    subplot(num_rows,num_cols,channel_index)
    plot( times, time_series(channel_index,:) )
    xlabel( sprintf('time (%s)',time_units) )
    ylabel(channel_units)
    if same_scale
        ylim(y_bounds)
    end
end

end