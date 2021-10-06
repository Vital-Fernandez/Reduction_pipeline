import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Rectangle


# Spectra folder
conf_file = '../reduction_conf.ini'
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
obsData = sr.loadConfData(conf_file)
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'imshift'
data_dict = {'reduc_tag': 'frame_shifted'}
frame_colors = {'Blue_arm': 'bone', 'Red_arm': 'gist_heat'}


class PlotFramePeak:

    def __init__(self, fits_address, color_frame, peak_coords=None, tol=20):

        # Attributes
        # self.night_conf = night_conf
        # self.conf_address = conf_address

        self.fits_address = fits_address
        self.fig, self.ax = None, None

        self.image_data = fits.getdata(fits_address)
        self.color = color_frame
        self.tol = tol
        self.max_cords = peak_coords

        # Plot the fits
        self.plot_frame(peak_coords)

        # Connect to event and show
        plt.tight_layout()
        plt.connect('button_press_event', self.on_click)
        plt.show()

    def plot_frame(self, center_coords=None):

        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(7, 10))

        zScale = ZScaleInterval()
        int_min, int_max = zScale.get_limits(self.image_data)
        self.ax.imshow(self.image_data, cmap=self.color, origin='lower', vmin=int_min, vmax=int_max,
                  interpolation='none', aspect='auto')

        if center_coords is not None:
            corner_coord = (center_coords - self.tol / 2).astype(int)

            self.ax.scatter(center_coords[0], center_coords[1], s=50, marker='+', color='yellow')
            self.ax.add_patch(Rectangle((corner_coord), self.tol, self.tol, fc='None', ec='yellow'))

            # Plot peak in region
            region_flux = self.image_data[corner_coord[1]:corner_coord[1] + self.tol,
                                          corner_coord[0]:corner_coord[0] + self.tol]

            local_peak_coords = np.unravel_index(np.argmax(region_flux, axis=None), region_flux.shape)

            peak_coords = corner_coord + local_peak_coords[::-1]
            self.ax.scatter(peak_coords[0], peak_coords[1], s=50, marker='o', color='red')

            self.ax.set_xlim(center_coords[0]-100, center_coords[0]+200)
            self.ax.set_ylim(center_coords[1]-100, center_coords[1]+200)

            # Store absolute max coordinates of that region
            self.max_cords = peak_coords

        return

    def on_click(self, event):

        if event.button == 3:

            if event.inaxes is not None:

                self.ax.cla()

                self.plot_frame()
                x, y = event.xdata, event.ydata
                event_coords = np.array([x, y]).astype(int)

                self.plot_frame(event_coords)

                plt.tight_layout()
                plt.draw()


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

    obj_shift_list = night_conf.get(f'obj_shift_list')

    if obj_shift_list is not None:

        for obj in obj_shift_list:

            for arm_color in colors:

                color_label = f'{arm_color}_arm'
                print(f'{obj} {color_label}')

                refLine_coor = night_conf[f'{obj}_{color_label}_ref_peak_array'].astype(int)

                idcs_shift = (pr.reducDf.frame_tag == obj) & \
                            (pr.reducDf.ISIARM == color_label) & \
                            (pr.reducDf.reduc_tag == 'cr_corr') & \
                            (pr.reducDf.valid_file)
                idcs_objs = pr.reducDf.loc[idcs_shift].index.values
                nFrames = idcs_objs.size

                # # Loop through the frames and select manually
                # shift_dict = {}
                # for i, idx_frame in enumerate(idcs_objs):
                #
                #     # Inputs
                #     obj_label = pr.reducDf.loc[idx_frame].frame_tag
                #     run = pr.reducDf.loc[idx_frame].RUN
                #     fits_address = f'{pr.reducDf.loc[idx_frame].file_location}/{pr.reducDf.loc[idx_frame].file_name}'
                #
                #     # Manual selection of peak and save coords to log
                #     peak_coords = night_conf.get(f'{run:.0f}_peak_coords')
                #     fits_plotter = PlotFramePeak(fits_address=fits_address,
                #                                  color_frame=frame_colors[color_label],
                #                                  peak_coords=peak_coords)
                #     night_conf[f'{run:.0f}_peak_coords'] = fits_plotter.max_cords.astype(int)
                #
                #     print(fits_plotter.max_cords)

                # Loop through the frames and compute the difference with respect to the first
                runs = pr.reducDf.loc[idcs_objs].RUN.values
                coords_first = night_conf[f'{runs[0]:.0f}_peak_coords'].astype(int)
                xShifts, yShifts = np.zeros(nFrames).astype(int), np.zeros(nFrames).astype(int)
                for i, run in enumerate(runs):
                    run_coords = night_conf[f'{run:.0f}_peak_coords'].astype(int)
                    xShifts[i] = (run_coords[0] - coords_first[0]) * -1
                    yShifts[i] = (run_coords[1] - coords_first[1]) * -1

                night_conf[f'{obj}_{arm_color}_xShift_array'] = xShifts.astype(int)
                night_conf[f'{obj}_{arm_color}_yShift_array'] = yShifts.astype(int)

                night_conf[f'{obj}_{color_label}_ref_peak_array'] = coords_first

                sr.safeConfData(conf_file, night_conf, section_name=night)

                # print('Frames being shifted by')
                # print(f'x: {xShifts}')
                # print(f'y: {yShifts}')
                # input("\nPress Enter to start task...")

                xShifts = night_conf[f'{obj}_{arm_color}_xShift_array']
                yShifts = night_conf[f'{obj}_{arm_color}_yShift_array']

                # Run the task
                for i, idx_frame in enumerate(idcs_objs):

                    # Inputs
                    obj_label = pr.reducDf.loc[idx_frame].frame_tag
                    run = pr.reducDf.loc[idx_frame].RUN
                    fits_address = f'{pr.reducDf.loc[idx_frame].file_location}/{pr.reducDf.loc[idx_frame].file_name}'

                    input_file = f'{pr.reducDf.loc[idx_frame].file_location}/{pr.reducDf.loc[idx_frame].file_name}'
                    output_file = input_file.replace('.fits', '_s.fits')

                    task_conf = {}
                    task_conf['color'] = arm_color
                    task_conf['run folder'] = run_folder
                    task_conf['input'] = input_file
                    task_conf['output'] = output_file
                    task_conf['xshift'] = xShifts[i]
                    task_conf['yshift'] = xShifts[i]

                    # Prepare iraf command
                    task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf)

                    # Log command
                    pr.store_command(task_name, command_log_address)

                    # Run the iraf command
                    pr.launch_command(task_name, task_conf_address)

                    # Log new files to DF
                    pr.object_to_dataframe(pr.task_attributes['output'], data_dict)

        # Generate pdf file
        output_file = f'{pr.reducFolders["reduc_data"]}/obj_shift'
        idcs_print = (pr.reducDf.reduc_tag == 'frame_shifted')
        pr.generate_step_pdf(idcs_print, file_address=output_file, ext=0, include_graph=True, sorting_pattern=['ISIARM', 'reduc_tag'],
                             plots_type='frame_combine', obs_conf=night_conf)

