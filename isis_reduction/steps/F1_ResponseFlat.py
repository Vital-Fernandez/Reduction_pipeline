import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
from pathlib import Path

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
obsData = sr.loadConfData('../reduction_conf.ini')
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'response'
data_dict = {'reduc_tag': 'response'}

# Loop through the nights
for night in obsData['data_location']['night_list']:

    print(f'\nDoing: {night}')

    # Establish night configuration
    pr = SpectraReduction(data_folder, obs_file=None)
    night_conf = obsData[night]
    night_folder = f'{data_folder}/{night}'
    objs, std_stars = night_conf['objects_list'], night_conf['standard_star_list']
    sky_objs = objs + std_stars

    pr.declare_catalogue(night_folder, objs=objs, std_stars=std_stars)
    run_folder = pr.reducFolders["flat lamp"]

    print(f'- Targets: {sky_objs}')

    for obj in sky_objs:

        for arm_color in colors:

            color_label = f'{arm_color}_arm'

            idcs_objFlat = (pr.reducDf.index == f'{obj}_{arm_color}_flat')

            # Inputs
            file_location = pr.reducDf[idcs_objFlat].file_location.values
            file_names = pr.reducDf[idcs_objFlat].file_name.values

            # In case one of the arm observations is missing:
            if len(file_names) == 1:

                input_files = file_location[0] + '/' + file_names[0]

                # Outputs
                output_files = input_files.replace('_flat', '_nflat')

                task_conf = {}
                task_conf['color'] = arm_color
                task_conf['run folder'] = run_folder
                task_conf['input'] = input_files
                task_conf['normalizing_flat'] = input_files
                task_conf['output'] = output_files
                task_conf['threshold'] = '3'
                task_conf['order'] = 30

                # Prepare iraf command
                task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf)

                output_path = Path(output_files)

                # Log command
                pr.store_command(task_name, command_log_address)

                # Run the iraf command
                pr.launch_command(task_name, task_conf_address)

                # Log new files to DF
                pr.object_to_dataframe(pr.task_attributes['output'], data_dict)

    #Generate pdf file
    output_file = f'{pr.reducFolders["reduc_data"]}/flat_normalization'
    idcs_print = (pr.reducDf.reduc_tag == 'response')
    pr.generate_step_pdf(idcs_print, file_address=output_file, ext=0, include_graph=True, sorting_pattern=['ISIARM', 'reduc_tag'])








#
