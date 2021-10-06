import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
from astropy.io import fits
import pandas as pd

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
conf_file = '../reduction_conf.ini'
obsData = sr.loadConfData(conf_file)
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'calibrate'
data_dict = {'reduc_tag': 'calibrate-nearest'}

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

    for obj in objs:

        for arm_color in colors:

            color_label = f'{arm_color}_arm'

            objIdx = (pr.reducDf.frame_tag == obj) & \
                        (pr.reducDf.reduc_tag == 'trace_spec') & \
                        (pr.reducDf.ISIARM == color_label) & \
                        (pr.reducDf.valid_file)

            if objIdx.sum() > 0:

                calibStar = night_conf[f'{obj}_{arm_color}_std']

                input_file = f'{pr.reducDf.loc[objIdx].file_name.values[0]}'
                sens_file = f'{run_folder}/{calibStar}_{arm_color}_sens.fits'
                output_file = f'{run_folder}/{input_file.replace(".fits", "")}' + f'_flux_nearest.fits'

                task_conf = {}
                task_conf['run folder'] = run_folder
                task_conf['color'] = arm_color
                task_conf['input'] = input_file
                task_conf['output'] = output_file
                task_conf['senstivityCurve'] = sens_file
                task_conf['airmass'] = fits.getval(f'{run_folder}/{input_file}', 'AIRMASS', 0)
                task_conf['exptime'] = fits.getval(f'{run_folder}/{input_file}', 'EXPTIME', 0)

                # Prepare iraf command
                task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf)

                # Log command
                pr.store_command(task_name, command_log_address)

                # Run the iraf command
                pr.launch_command(task_name, task_conf_address)

                # Log new files to DF
                pr.object_to_dataframe(pr.task_attributes['output'], data_dict)

# indeces_print = (dz.reducDf.reduc_tag == 'flux_calibrated_objects_flocal') | (dz.reducDf.reduc_tag == 'flux_calibrated_objects_fglobal')
# dz.generate_step_pdf(indeces_print, file_address = dz.reducFolders['reduc_data'] + 'calibrated_objects', plots_type = 'spectra', ext = 0)
#
# print 'Data treated'


