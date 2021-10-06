import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
from x_cosmics import cosmicsimage, fromfits
from matplotlib import pyplot as plt, patches
from astropy.io import fits
from astropy.visualization import ZScaleInterval

def on_click(event):

    if event.button == 3:

        if event.inaxes is not None:

            # Clear previous data
            plt.cla()

            # Plot the image
            ax.imshow(image_data, cmap=frame_colors[colors_arm[i]], origin='lower', vmin=int_min, vmax=int_max,
                      interpolation='nearest', aspect='auto')

            # Get event data
            x, y = event.xdata, event.ydata
            x_cords, y_cords = int(x + 0), int(y + 0)
            ax.scatter(x_cords, y_cords, s=50, edgecolor='yellow', facecolor='none')

            # Plot max locatio
            max_values = np.max(image_data)
            max_indeces_sec = np.where(image_data == max_values)
            ax.scatter(max_indeces_sec[1], max_indeces_sec[0], s=30, facecolor='red')

            # Plot local maxima
            section = image_data[y_cords - 5:y_cords + 5, x_cords - 5:x_cords + 5]
            max_value_sec = np.max(section)
            max_indeces_sec = np.where(image_data == max_value_sec)
            x_max, y_max = max_indeces_sec[1][0], max_indeces_sec[0][0]
            ax.scatter(x_max, y_max, s=40, facecolor='black', edgecolor='yellow')

            mark_label = f'{obj}_{colors_arm[i]}_ref_peak_array'
            peak_array = np.array([x_cords, y_cords]).astype(int)
            print(f'- New value: {peak_array}')
            night_conf[mark_label] = peak_array

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

        mark_label = f'{obj}_{colors_arm[i]}_ref_peak_array'
        if mark_label in night_conf:
            peak_array = night_conf[mark_label].astype(int)
            ax.scatter(peak_array[0], peak_array[1], s=50, facecolor='green')
            print(f'- Stored value: {peak_array}')

        max_values = np.max(image_data)
        max_indeces_sec = np.where(image_data == max_values)
        ax.scatter(max_indeces_sec[1], max_indeces_sec[0], s=15, facecolor='red')
        ax.set_title(f'{night}: {obj}')

        # Display the interactive plot
        plt.tight_layout()
        plt.connect('button_press_event', on_click)
        plt.show()

        sr.safeConfData(conf_file, night_conf, section_name=night)

