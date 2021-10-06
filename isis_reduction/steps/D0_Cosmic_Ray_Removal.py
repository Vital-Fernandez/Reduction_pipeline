# import os
# import pyfits
# from DZ_observation_reduction import spectra_reduction

import os
import src.specsiser as sr
from pipeline import SpectraReduction
from astropy.io import fits
from x_cosmics import cosmicsimage, fromfits
import lacosmic

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
obsData = sr.loadConfData('../reduction_conf.ini')
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'flatcombine'
data_dict = {'reduc_tag': 'cr_corr'}

# Loop through the nights
for night in obsData['data_location']['night_list']:

    print(f'\nDoing: {night}')

    # Establish night configuration
    pr = SpectraReduction(data_folder, obs_file=None)
    night_conf = obsData[night]
    night_folder = f'{data_folder}/{night}'
    objs, std_stars = night_conf['objects_list'], night_conf['standard_star_list']

    pr.declare_catalogue(night_folder, objs=objs, std_stars=std_stars)

    sky_objs = objs + std_stars
    index_object = (pr.reducDf.frame_tag.isin(sky_objs)) & (pr.reducDf.reduc_tag == 'biascorr') & pr.reducDf.valid_file
    Files_Folder = pr.reducDf.loc[index_object, 'file_location'].values
    Files_Name = pr.reducDf.loc[index_object, 'file_name'].values
    objects = pr.reducDf.loc[index_object, 'frame_tag'].values
    colors_arm = pr.reducDf.loc[index_object, 'ISIARM'].values

    for i, obj in enumerate(objects):

        print(f'\n-Treating: {obj} ({Files_Name[i]}) {i}/{Files_Folder.size}\n, {colors_arm[i]}')

        # Get the data ready for the task
        fits_file = f'{Files_Folder[i]}/{Files_Name[i]}'
        fitsdata, hdr = fromfits(fits_file, verbose=False)

        # Get the object configuration values:
        if obj in std_stars:
            if 'star_sigclip' in night_conf:
                sigclip = float(night_conf['star_sigclip'])
            else:
                sigclip = 15
        else:
            if 'obj_sigclip' in night_conf:
                sigclip = float(night_conf['obj_sigclip'])
            else:
                sigclip = 15

        gain = fits.getval(fits_file, 'GAIN', 0)
        readnoise = fits.getval(fits_file, 'READNOIS', 0)
        lacosmic_param = [gain, readnoise, sigclip]

        print(f'-- Parameters: gain {gain}, readnoise {readnoise}, sigclip {sigclip}')

        # Frame cosmic object
        c = cosmicsimage(fitsdata, gain=lacosmic_param[0], readnoise=lacosmic_param[1], satlevel=70000, sigclip=lacosmic_param[2])

        # Run the fitting
        c.run(maxiter=4)

        # Write the cleaned image into a new FITS file, conserving the original header :
        output_clean = f'{Files_Folder[i]}/{Files_Name[i][0:Files_Name[i].rfind(".")]}_cr.fits'
        output_mask = f'{Files_Folder[i]}/{Files_Name[i][0:Files_Name[i].rfind(".")]}_mask.fits'

        # Delete the file if it already exists
        if os.path.isfile(output_clean):
            os.remove(output_clean)
        if os.path.isfile(output_mask):
            os.remove(output_mask)

        # Store the frames
        hdu_clean = fits.PrimaryHDU(c.cleanarray.transpose(), hdr)
        hdu_clean.writeto(output_clean)

        masked_array = c.mask.transpose().astype(int)
        hdu_mask = fits.PrimaryHDU(masked_array, hdr)
        hdu_mask.writeto(output_mask)

        # Add objects to data frame with the new frame_tag
        pr.object_to_dataframe(output_clean, data_dict)

    # Generate pdf file
    output_file = f'{pr.reducFolders["reduc_data"]}/science_cosmic_removal'
    idcs_print = (pr.reducDf.reduc_tag == 'cr_corr') & (pr.reducDf.frame_tag.isin(objs))
    pr.generate_step_pdf(idcs_print, file_address=output_file, ext=0, plots_type='cosmic_removal')

    output_file = f'{pr.reducFolders["reduc_data"]}/stdstars_cosmic_removal'
    idcs_print = (pr.reducDf.reduc_tag == 'cr_corr') & (pr.reducDf.frame_tag.isin(std_stars))
    pr.generate_step_pdf(idcs_print, file_address=output_file, ext=0, plots_type='cosmic_removal')

