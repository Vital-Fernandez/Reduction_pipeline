import os
import numpy as np
import pandas as pd
import src.specsiser as sr
from pipeline import SpectraReduction

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
obsData = sr.loadConfData('../reduction_conf.ini')
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'flatcombine'
data_dict = {'reduc_tag': 'flatcombine'}

# No warning of modifying slice
pd.options.mode.chained_assignment = None

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

            idcs_obj_arm = (pr.reducDf.frame_tag == obj) &\
                           (pr.reducDf.ISIARM == color_label) &\
                           (pr.reducDf.reduc_tag == 'cr_corr')

            cenwave_array = pr.reducDf.loc[idcs_obj_arm].CENWAVE.values
            num_cenwaves = np.unique(cenwave_array).size
            assert num_cenwaves < 2, 'WARNING More than one object configuration'

            # In case one of the arm observations is missing:
            if num_cenwaves == 1:

                idcs_flats = (pr.reducDf.frame_tag == 'flat') & \
                             (pr.reducDf.ISIARM == color_label) & \
                             (pr.reducDf.reduc_tag == 'biascorr')

                # Trick to avoid frames with no CENWAVES
                df_flats = pr.reducDf.loc[idcs_flats]
                df_flats['CENWAVE'] = df_flats['CENWAVE'].astype(float)
                idcs_combFlats = df_flats['CENWAVE'].apply(np.isclose, b=float(cenwave_array[0]), atol=3)

                # Inputs
                file_location = df_flats[idcs_combFlats].file_location.values
                file_names = df_flats[idcs_combFlats].file_name.values
                input_array = file_location + '/' + file_names
                input_list_name = f'{obj}_flatCombine_{arm_color}_file.list'

                # Outputs
                output_file_name = f'{obj}_{arm_color}_flat.fits'
                print(f'{obj} {arm_color}: ', len(file_names))

                #Set the task attributes
                task_conf = {}

                task_conf['color'] = arm_color
                task_conf['input array'] = input_array
                task_conf['run folder'] = run_folder
                task_conf['in_list_name'] = input_list_name
                task_conf['input'] = f'@{run_folder}/{input_list_name}'
                task_conf['output'] = f'{run_folder}/{output_file_name}'
                task_conf['combine'] = 'median'
                task_conf['reject'] = 'crreject'
                task_conf['scale'] = 'mode'
                task_conf['ccdtype'] = '""'
                task_conf['gain'] = 'gain'
                task_conf['snoise'] = 'readnois'

                # Prepare iraf command
                task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf)

                # Log command
                pr.store_command(task_name, command_log_address)

                # Run the iraf command
                pr.launch_command(task_name, task_conf_address)

                # Log new files to DF
                pr.object_to_dataframe(pr.task_attributes['output'], data_dict)

    #Generate pdf file
    output_file = f'{pr.reducFolders["reduc_data"]}/flat_combines'
    idcs_print = (pr.reducDf.reduc_tag == 'flatcombine')
    pr.generate_step_pdf(idcs_print, file_address=output_file, ext=0, include_graph=True, sorting_pattern=['ISIARM', 'reduc_tag'])



