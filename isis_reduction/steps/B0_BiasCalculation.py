import src.specsiser as sr
from pipeline import SpectraReduction
from astropy.io import fits

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
obsData = sr.loadConfData('../reduction_conf.ini')
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'zerocombine'
data_dict = {'reduc_tag': 'master_bias'}

# Loop through the nights
for night in obsData['data_location']['night_list']:

    print(f'\nDoing: {night}\n')

    # Establish night configuration
    night_conf = obsData[night]
    night_folder = f'{data_folder}/{night}'
    objs, std_stars = night_conf['objects_list'], night_conf['standard_star_list']

    pr = SpectraReduction(data_folder, obs_file=None)
    pr.declare_catalogue(night_folder, objs=objs, std_stars=std_stars)

    run_folder = pr.reducFolders["bias"]

    for arm_color in colors:

        # Arm label
        color_label = f'{arm_color}_arm'

        # Get the files to treat
        idcs = (pr.reducDf.reduc_tag == 'bias') & (pr.reducDf.ISIARM == color_label) & pr.reducDf.valid_file
        list_file_names = pr.reducDf.loc[idcs, 'file_name'].values

        # Task configuration
        task_conf = {}

        task_conf['input array'] = pr.reducFolders['raw data'] + '/' + list_file_names + '[1]'
        task_conf['color'] = arm_color
        task_conf['run folder'] = f'{run_folder}'
        task_conf['in_list_name'] = f'bias_{arm_color}'
        task_conf['input'] = f'@{run_folder}/{task_conf["in_list_name"]}'
        task_conf['output'] = f'{run_folder}/master_bias_{arm_color}.fits'
        task_conf['combine'] = 'median'

        # Prepare iraf command
        task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf)

        # Log command
        pr.store_command(task_name, command_log_address)

        # Run the iraf command
        pr.launch_command(task_name, task_conf_address, verbose=False)

        # Log new files to DF
        pr.object_to_dataframe(pr.task_attributes['output'], data_dict)

    #Display biassec, trimsec data
    idcs_print = (pr.reducDf.reduc_tag == 'master_bias')

    # list_mBias = pr.reducDf.loc[idcs_print].file_location.values + '/' + pr.reducDf.loc[idcs_print].file_name.values
    # for bias_file in list_mBias:
    #     print(fits.getval(bias_file, 'ISIARM', 0))
    #     print('BIASSEC', fits.getval(bias_file, 'BIASSEC', 0))
    #     print('TRIMSEC', fits.getval(bias_file, 'TRIMSEC', 0)),'\n'

    #Generating pdf output
    output_file = f'{pr.reducFolders["reduc_data"]}/global_bias'
    pr.generate_step_pdf(idcs_print, file_address=output_file, plots_type='fits_compare', ext=0, include_graph=True)
   
