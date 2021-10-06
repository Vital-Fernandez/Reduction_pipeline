import os
import src.specsiser as sr
from pipeline import SpectraReduction

  
# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
obsData = sr.loadConfData('../reduction_conf.ini')
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'transform'
data_dict = {'reduc_tag': 'arcs_wave_calibrated'}

# Loop through the nights
for night in obsData['data_location']['night_list']:

    # Establish night configuration
    pr = SpectraReduction(data_folder, obs_file=None)
    night_conf = obsData[night]
    night_folder = f'{data_folder}/{night}'
    objs, std_stars = night_conf['objects_list'], night_conf['standard_star_list']

    pr.declare_catalogue(night_folder, objs=objs, std_stars=std_stars)
    run_folder = pr.reducFolders["arcs"]

    # Loop through the arms
    for arm_color in colors:

        color_label = f'{arm_color}_arm'

        idx_arcs = (pr.reducDf.reduc_tag == 'biascorr') & (pr.reducDf.ISIARM == color_label) &\
                   (pr.reducDf.frame_tag == 'arc') & pr.reducDf.valid_file

        File_Folders = pr.reducDf.loc[idx_arcs, 'file_location'].values
        File_Names = pr.reducDf.loc[idx_arcs, 'file_name'].values

        for i in range(len(File_Names)):

            # Transform the arc
            output_file = File_Names[i].replace(".fits", "_w.fits")

            task_conf = {}
            task_conf['run folder'] = run_folder
            task_conf['color'] = arm_color
            task_conf['input'] = f'{File_Names[i]}'
            task_conf['output'] = f'{output_file}'
            task_conf['database'] = f'database'
            task_conf['fitnames'] = f'{os.path.splitext(File_Names[i])[0]}'

            # Prepare iraf command
            task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf)

            # Log command
            pr.store_command(task_name, command_log_address)

            # Run command
            pr.launch_command(task_name, task_conf_address, verbose=True)

            # Log new files to DF
            pr.object_to_dataframe(f'{File_Folders[i]}/{output_file}', data_dict)

    # #Generate pdf file
    # idcs_print = (pr.reducDf.reduc_tag == 'arcs_wave_calibrated')
    # output_file = f'{pr.reducFolders["reduc_data"]}/arcs_wave_calibrated'
    # pr.generate_step_pdf(idcs_print, file_address=output_file, verbose=True, ext=0, plots_type='arcs_calibration')
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


