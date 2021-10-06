import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
import os
# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
obsData = sr.loadConfData('../reduction_conf.ini')
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'ccdproc'
data_dict = {'reduc_tag': 'flat_corr'}

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

        idcs_objs = ((pr.reducDf.frame_tag.isin(objs) & (pr.reducDf.reduc_tag == 'obj_comb')) |
                     (pr.reducDf.frame_tag.isin(std_stars) & (pr.reducDf.reduc_tag == 'cr_corr'))) & \
                     (pr.reducDf.ISIARM == color_label) & \
                     (pr.reducDf.valid_file)
        target_indeces = pr.reducDf.loc[idcs_objs].index.values
        #
        #
        # target_file_list = pr.reducDf.loc[idcs_objs].index.values
        # target_label_list = pr.reducDf.loc[idcs_objs].frame_tag.values

        for idx in target_indeces:

            obj_label = pr.reducDf.loc[idx].frame_tag
            input_file = f'{pr.reducDf.loc[idx].file_location}/{pr.reducDf.loc[idx].file_name}'
            flat_file = f'{pr.reducFolders["flat lamp"]}/{obj_label}_{arm_color}_nflat.fits'
            output_name = f'{pr.reducDf.loc[idx].file_location}/{obj_label}_{arm_color}_f.fits'

            task_conf = {}
            task_conf['color'] = arm_color
            task_conf['run folder'] = run_folder
            task_conf['input'] = input_file
            task_conf['output'] = output_name
            task_conf['flatcor'] = 'yes'
            task_conf['flat'] = flat_file

            # Prepare iraf command
            task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf)

            # Log command
            pr.store_command(task_name, command_log_address)

            # Run the iraf command
            pr.launch_command(task_name, task_conf_address)

            # Log new files to DF
            pr.object_to_dataframe(pr.task_attributes['output'], data_dict)

    #Generate pdf file
    output_file = f'{pr.reducFolders["reduc_data"]}/obj_flat_corr'
    idcs_print = (pr.reducDf.reduc_tag == 'flat_corr')
    pr.generate_step_pdf(idcs_print, file_address=output_file, ext=0, include_graph=True, sorting_pattern=['ISIARM', 'reduc_tag'])

