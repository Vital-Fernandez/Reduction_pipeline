import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
from astropy.io import fits

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
obsData = sr.loadConfData('../reduction_conf.ini')
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'imcombine'
data_dict = {'reduc_tag': 'obj_comb'}

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
    objShift_list = night_conf.get('obj_shift_list', [])

    for obj in objs:

        reduc_tag = 'frame_shifted' if obj in objShift_list else 'cr_corr'

        for arm_color in colors:

            color_label = f'{arm_color}_arm'

            idcs_objs = (pr.reducDf.frame_tag == obj) & \
                        (pr.reducDf.ISIARM == color_label) & \
                        (pr.reducDf.reduc_tag == reduc_tag) & \
                        (pr.reducDf.valid_file)

            files_folder = pr.reducDf.loc[idcs_objs, 'file_location'].values
            files_name = pr.reducDf.loc[idcs_objs, 'file_name'].values
            n_files = len(files_name)

            if n_files > 0:

                # Inputs
                input_array = files_folder + '/' + files_name
                input_list_name = f'{obj}_imCombine_{arm_color}_file.list'

                # Outputs
                output_file_name = f'{obj}_{arm_color}.fits'

                # Compute effective air mass and median flux scale
                bg_region = night_conf[f'{obj}_{color_label}_scale_array'].astype(int)
                air_mass_array = np.zeros(n_files)
                scale_array = np.zeros(n_files)
                for i, file_address in enumerate(input_array):
                    frame_data = fits.getdata(file_address, 0)
                    air_mass_array[i] = fits.getval(file_address, 'AIRMASS', 0)
                    scale_array[i] = np.median(frame_data[bg_region[2]:bg_region[3], bg_region[0]:bg_region[1]])
                air_eff = np.mean(air_mass_array)
                print(f'- {obj} {arm_color}: air mass {air_mass_array.mean():0.3f}+/-{air_mass_array.std():0.3f},'
                      f' {scale_array.mean():0.3f}+/-{scale_array.std():0.3f}')

                #Set the task attributes
                task_conf = {}

                task_conf['color'] = arm_color
                task_conf['run folder'] = run_folder
                task_conf['input array'] = input_array
                task_conf['in_list_name'] = input_list_name
                task_conf['input'] = f'@{run_folder}/{input_list_name}'
                task_conf['output'] = f'{run_folder}/{output_file_name}'

                task_conf['combine'] = 'median'
                task_conf['scale'] = 'median'
                task_conf['statsec'] = '[{XA}:{XB},{YA}:{YB}]'.format(XA=bg_region[0], XB=bg_region[1],
                                                                      YA=bg_region[2], YB=bg_region[3])
                task_conf['reject'] = 'crreject'
                task_conf['weight'] = '""'
                task_conf['gain'] = 'GAIN'
                task_conf['snoise'] = 'READNOIS'

                # Prepare iraf command
                task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf)

                # Log command
                pr.store_command(task_name, command_log_address)

                # Run the iraf command
                pr.launch_command(task_name, task_conf_address)

                # Log new files to DF
                pr.object_to_dataframe(pr.task_attributes['output'], data_dict)

    #Generate pdf file
    output_file = f'{pr.reducFolders["reduc_data"]}/obj_combined'
    idcs_print = (pr.reducDf.reduc_tag == 'obj_comb') | \
                 ((pr.reducDf.reduc_tag == 'cr_corr') & (pr.reducDf.frame_tag.isin(objs)) & (~pr.reducDf.frame_tag.isin(objShift_list))) | \
                 (pr.reducDf.reduc_tag == 'frame_shifted')

    pr.generate_step_pdf(idcs_print, file_address=output_file, ext=0, include_graph=True, sorting_pattern=['ISIARM', 'reduc_tag'],
                         plots_type='frame_combine', obs_conf=night_conf)
