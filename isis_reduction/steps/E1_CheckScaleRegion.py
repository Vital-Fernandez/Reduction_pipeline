import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
from x_cosmics import cosmicsimage, fromfits
from matplotlib import pyplot as plt, patches
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib.widgets import RectangleSelector
import pandas as pd

def line_select_callback(eclick, erelease):

    # Clear previous data
    plt.cla()

    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    scale_array = np.array([x1, x2, y1, y2]).astype(int)

    # Plot the image
    ax.imshow(image_data, cmap=frame_colors[colors_arm[i]], origin='lower', vmin=int_min, vmax=int_max,
              interpolation='nearest', aspect='auto')

    # Get event data
    ax.add_patch(patches.Rectangle((scale_array[0], scale_array[2]),
                                   scale_array[1] - scale_array[0],
                                   scale_array[3] - scale_array[2],
                                   linewidth=2, color='black', fill=False))

    mark_label = f'{obj}_{colors_arm[i]}_scale_array'
    print(f'- New value: {scale_array}')
    night_conf[mark_label] = scale_array

    ax.set_title(f'{night}: {obj}')

    plt.tight_layout()
    plt.draw()



# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
conf_file = '../reduction_conf.ini'
obsData = sr.loadConfData(conf_file)
command_log_address = obsData['data_location']['command_log_location']

frame_colors = {'Blue_arm': 'bone', 'Red_arm': 'gist_heat'}

# Loop through the nights
for night in obsData['data_location']['night_list']:

    print(f'\nDoing: {night}')

    # Establish night configuration
    pr = SpectraReduction(data_folder, obs_file=None)
    night_conf = obsData[night]
    night_folder = f'{data_folder}/{night}'
    objs, std_stars = night_conf['objects_list'], night_conf['standard_star_list']

    pr.declare_catalogue(night_folder, objs=objs, std_stars=std_stars)

    index_object = (pr.reducDf.frame_tag.isin(objs)) & (pr.reducDf.reduc_tag == 'cr_corr') & pr.reducDf.valid_file
    Files_Folder = pr.reducDf.loc[index_object, 'file_location'].values
    Files_Name = pr.reducDf.loc[index_object, 'file_name'].values
    objects = pr.reducDf.loc[index_object, 'frame_tag'].values
    colors_arm = pr.reducDf.loc[index_object, 'ISIARM'].values

    for i, obj in enumerate(objects):

        print(f'\n-Treating: {obj} ({Files_Name[i]}) {i}/{Files_Folder.size}')

        # Get the data ready for the task
        fits_file = f'{Files_Folder[i]}/{Files_Name[i]}'

        with fits.open(fits_file) as hdu_list:
            image_data = hdu_list[0].data

        fig, ax = plt.subplots(1, 1, figsize=(7, 10))

        # Plotting fits
        zScale = ZScaleInterval()
        int_min, int_max = zScale.get_limits(image_data)

        ax.imshow(image_data, cmap=frame_colors[colors_arm[i]], origin='lower', vmin=int_min, vmax=int_max,
                  interpolation='nearest', aspect='auto')

        mark_label = f'{obj}_{colors_arm[i]}_scale_array'
        if mark_label in night_conf:
            scale_array = night_conf[mark_label].astype(int)
            ax.add_patch(patches.Rectangle((scale_array[0], scale_array[2]),
                                           scale_array[1] - scale_array[0],
                                           scale_array[3] - scale_array[2],
                                           linewidth=2, color='black', fill=False))

        ax.set_title(f'{night}: {obj}')

        # Display the interactive plot
        plt.tight_layout()

        rs = RectangleSelector(ax, line_select_callback,
                               drawtype='box', useblit=False, button=[1],
                               minspanx=5, minspany=5, spancoords='pixels',
                               interactive=True)

        plt.show()

        sr.safeConfData(conf_file, night_conf, section_name=night)


