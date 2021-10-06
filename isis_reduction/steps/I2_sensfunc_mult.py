import os
import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
from astropy.io import fits
from shutil import copyfile

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
conf_file = '../reduction_conf.ini'
obsData = sr.loadConfData(conf_file)
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'sensfunc'
data_dict = {'reduc_tag': 'sensfunc'}

# Loop through the nights
for night in obsData['data_location']['night_list']:

    print(f'\nDoing: {night}')

    # Establish night configuration
    pr = SpectraReduction(data_folder, obs_file=None)
    night_conf = obsData[night]
    night_folder = f'{data_folder}/{night}'
    objs, std_stars = night_conf['objects_list'], night_conf['standard_star_list']

    pr.declare_catalogue(night_folder, objs=objs, std_stars=std_stars)
    run_folder = pr.reducFolders['objects']

    for arm_color in colors:

        color_label = f'{arm_color}_arm'

        night_cenWaves = night_conf[f'night_{arm_color}_cenwaves_array'].astype(int)

        # List of global std stars
        std_global_file_list = np.char.add(np.char.add(f'{run_folder}/', night_cenWaves.astype(str)), f'_{arm_color}_std.txt')

        for std_file in std_global_file_list:

            sens_file = std_file.replace('_std.txt', '_sens.fits')

            task_conf = {}
            task_conf['run folder'] = run_folder
            task_conf['color'] = arm_color
            task_conf['input'] = std_file
            task_conf['output'] = sens_file
            task_conf['functio'] = 'spline3'
            task_conf['order'] = 10
            task_conf['graphs'] = 'si'

            # Prepare iraf command
            task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf, overwrite=False)

            # Log command
            pr.store_command(task_name, command_log_address)

            # # Run the iraf command
            # pr.launch_command(task_name, task_conf_address)
            #
            # # Log new files to DF
            # pr.object_to_dataframe(pr.task_attributes['output'], data_dict)

