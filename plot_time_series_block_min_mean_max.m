function f = plot_time_series_block_min_mean_max(plot_times,time_series,rls_start_time,rls_stop_time,supervisor_start_times,ts_name)
%UNTITLED12 Summary of this function goes here
%   Detailed explanation goes here

num_plot_points = numel(plot_times);
[min_over_regions,mean_regions,max_over_regions] = make_min_mean_max_of_blocks(time_series,num_plot_points,1);
bar_down = mean_regions-min_over_regions;
bar_up = max_over_regions-mean_regions;
y_bounds = [ min(min_over_regions) max(max_over_regions) ];
f = figure('Position',[0 0 1000 500]);
hold on
errorbar( plot_times, mean_regions, bar_down, bar_up, 'Color', 'k' )
plot([rls_start_time rls_start_time],y_bounds,':g','LineWidth',5)
plot([rls_stop_time rls_stop_time],y_bounds,':r','LineWidth',5)
plot(  supervisor_start_times, zeros( 1, numel(supervisor_start_times) ), 'ob'  )
ylim(y_bounds)
hold off
legend({sprintf('min-mean-max %s over regions',ts_name), 'RLS start time', 'RLS stop time','supervisor restarts'},'Location','northeastoutside')
xlabel('Time (ms)')
ylabel(ts_name)

end