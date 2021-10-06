from os import path
from pipeline import SpectraReduction
import src.specsiser as sr
import numpy as np

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'

# Load tools
obsData = sr.loadConfData('../reduction_conf.ini')

#Extensions
ext = 1

# Loop through the nights
for night in obsData['data_location']['night_list']:

    print(f'\nDoing: {night}')

    # Establish night configuration
    night_conf = obsData[night]
    night_folder = f'{data_folder}/{night}'
    objs, std_stars = night_conf['objects_list'], night_conf['standard_star_list']

    pr = SpectraReduction(data_folder, obs_file=None)
    pr.declare_catalogue(night_folder, objs=objs, std_stars=std_stars)

    #-- Bias frames
    output_list_file = f'{pr.reducFolders["reduc_data"]}/obs_bias.list'
    output_pdf_file = f'{pr.reducFolders["reduc_data"]}/bias_frames'

    indeces = (pr.reducDf.frame_tag == 'bias') & (pr.reducDf.file_location.str.contains('raw_fits'))
    list_files = pr.reducDf.loc[indeces, 'file_name'].values
    np.savetxt(output_list_file, list_files, fmt='%s')
    pr.generate_step_pdf(indeces, file_address=output_pdf_file, include_graph=True, verbose=False, ext=ext)
    print('Bias pdf generated')

    #-- Flat frames
    output_list_file = f'{pr.reducFolders["reduc_data"]}/obs_flat.list'
    output_pdf_file = f'{pr.reducFolders["reduc_data"]}/flat_frames'

    indeces  = (pr.reducDf.frame_tag == 'flat') & (pr.reducDf.file_location.str.contains('raw_fits'))
    list_files = pr.reducDf.loc[indeces, 'file_name'].values
    np.savetxt(output_list_file, list_files, fmt='%s')
    pr.generate_step_pdf(indeces, file_address=output_pdf_file, include_graph=True, verbose=False, ext=ext)
    print('Flat pdf generated')

    #-- Sky frames
    output_list_file = f'{pr.reducFolders["reduc_data"]}/obs_sky.list'
    output_pdf_file = f'{pr.reducFolders["reduc_data"]}/sky_frames'

    indeces  = (pr.reducDf.frame_tag == 'sky') & (pr.reducDf.file_location.str.contains('raw_fits'))
    list_files = pr.reducDf.loc[indeces, 'file_name'].values
    np.savetxt(output_list_file, list_files, fmt='%s')
    pr.generate_step_pdf(indeces, file_address=output_pdf_file, include_graph=True, verbose=False, ext=ext)
    print('Sky pdf generated')

    #-- Arc frames
    output_list_file = f'{pr.reducFolders["reduc_data"]}/obs_arc.list'
    output_pdf_file = f'{pr.reducFolders["reduc_data"]}/arc_frames'

    indeces = (pr.reducDf.frame_tag == 'arc')  & (pr.reducDf.file_location.str.contains('raw_fits'))
    list_files = pr.reducDf.loc[indeces, 'file_name'].values
    np.savetxt(output_list_file, list_files, fmt='%s')
    pr.generate_step_pdf(indeces, file_address=output_pdf_file, include_graph=True, verbose=False, ext=ext)
    print('Arc pdf generated')

    #--target frames
    output_list_file = f'{pr.reducFolders["reduc_data"]}/obs_target.list'
    output_pdf_file = f'{pr.reducFolders["reduc_data"]}/targets_frames'

    indeces = (pr.reducDf.OBSTYPE == 'TARGET') & (pr.reducDf.file_location.str.contains('raw_fits'))
    list_files = pr.reducDf.loc[indeces, 'file_name'].values
    np.savetxt(output_list_file, list_files, fmt='%s')
    pr.generate_step_pdf(indeces, file_address=output_pdf_file, verbose=False, ext=ext)
    print('Target pdf generated')

    print('All documents generated')






