import numpy as np
import pylatex

from numpy.polynomial.legendre  import legval
from matplotlib import pyplot as plt, patches

from itertools import cycle, zip_longest
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from pylatex import Document, Package, Figure, NoEscape

def grouper(n, iterable, padvalue=None):
  "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
  return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

class fits_plots():

    def __init__(self):

        self.fig_type = None
        self.page_grid_width = None
        self.page_grid_height = None

        self.Blue_Arc_lines = [4426.2047,
                               4657.8615,
                               4726.87,
                               4735.91,
                               5400.5615,
                               5852.5224,
                               6402.25,
                               6717.0216,
                               7723.8
                               ]

        self.Red_Arc_lines = [7245.17,
                              7383.98,
                              7635.1,
                              7948.18,
                              8006.16,
                              8014.79,
                              8115.31,
                              9122.9588,
                              9657.706,
                              9784.3944,
                              10052.1,
                              10470.05
                              ]

        linestyles = ["-", "-", "--", "--", "-.", "-.", ":", ":"]
        self.linecycler = cycle(linestyles)

        self.plots_dict = {'Blue': 0, 'Red': 1}
        self.frames_colors = {'Blue_arm': 'bone', 'Red_arm': 'gist_heat'}

    def fits_to_frame(self, frame_address, axis_plot, color=None, ext=0, valid=True, image_data=None, section=None):

        if color not in self.frames_colors:
            cmap_frame = 'gray'
        else:
            cmap_frame = self.frames_colors[color]

        # Clear the axis from previous plot
        axis_plot.clear()
        axis_plot.axes.get_yaxis().set_visible(False)
        axis_plot.axes.get_xaxis().set_visible(False)

        # Open the image
        if frame_address is not None:
            with fits.open(frame_address) as hdu_list:
                image_data = hdu_list[ext].data

        # Complete image plotted
        if section is None:

            # Get zscale limits for plotting the image
            IntensityLimits = ZScaleInterval()
            int_min, int_max = IntensityLimits.get_limits(image_data)[0], IntensityLimits.get_limits(image_data)[1]

            # Plot the data
            axis_plot.imshow(image_data, cmap=cmap_frame, origin='lower', vmin=int_min, vmax=int_max,
                             interpolation='nearest', aspect='auto')
            axis_plot.set_xlim(0, image_data.shape[1])
            axis_plot.set_ylim(0, image_data.shape[0])

        # Only a section
        else:

            x_max, y_max = int(section[0]), int(section[1])
            max_region = image_data[y_max - 5:y_max + 5:, x_max - 5:x_max + 5]

            # Get zscale limits for plotting the image
            IntensityLimits = ZScaleInterval()
            int_min, int_max = IntensityLimits.get_limits(max_region)[0], IntensityLimits.get_limits(max_region)[1]

            # Plot the data
            axis_plot.imshow(max_region, cmap=cmap_frame, origin='lower', vmin=int_min, vmax=int_max,
                             interpolation='nearest', aspect='auto')
            axis_plot.set_xlim(0, max_region.shape[1])
            axis_plot.set_ylim(0, max_region.shape[0])

        return

    def plot_spectral_axis(self, frame_address, axis_plot, label, ext=0, valid=True):

        with fits.open(frame_address) as hdu_list:
            image_data = hdu_list[ext].data

        line_Style = '-' if valid else ':'
        # colorTitle = 'black' if validity_check[index_i] else 'red'

        y_values = image_data.mean(axis=1)
        x_values = range(len(y_values))
        axis_plot.plot(x_values, y_values, label=label, linestyle=line_Style)

        return

    def compute_background_median(self, fits_address, trace_address):

        # Read the text file
        file_trace = open(trace_address)
        file_lines = file_trace.readlines()
        file_trace.close()

        # Open the image
        with fits.open(fits_address) as hdu_list:
            image_data = hdu_list[0].data

        # Reference indeces
        idx_start = file_lines.index('\taxis\t1\n') + 1  # That '\t1' is the orientation of the frame
        idx_match = [i for i in range(len(file_lines)) if '\tcenter\t' in file_lines[i]]

        # Aperture data
        center_line = file_lines[idx_match[0]]
        lower_line = file_lines[idx_match[0] + 1]
        upper_line = file_lines[idx_match[0] + 2]
        aper_center = map(float, center_line.split()[1:])
        aper_low = map(float, lower_line.split()[1:])
        aper_high = map(float, upper_line.split()[1:])
        number_pixels = aper_high - aper_low
        number_pixels, int(number_pixels)

        # Fitting coefficients
        coef_n = file_lines[idx_start].split()[1]
        fit_type = float(file_lines[idx_start + 1].split()[0])
        order = int(float(file_lines[idx_start + 2].split()[0]))
        # xmin        = int(float(file_lines[idx_start + 3].split()[0]))
        xmin = 0
        xmax = int(float(file_lines[idx_start + 4].split()[0]))
        coefs = np.empty(order)
        for i in range(len(coefs)):
            coefs[i] = float(file_lines[idx_start + 5 + i].split()[0])

        # Plot the polynomial
        y_range = np.arange(float(xmin), float(xmax))
        n = (2 * y_range - (xmax + xmin)) / (xmax - xmin)
        poly_leg = np.legval(n, coefs)
        trace_curve = poly_leg + aper_center[0]

        # Plot Background region
        idx_background = [i for i in range(len(file_lines)) if '\t\tsample' in file_lines[i]]
        background_line = file_lines[idx_background[0]].split()[1:]

        y_grid, x_grid = np.mgrid[0:image_data.shape[0], 0:image_data.shape[1]]

        # GridAxis.imshow(image_data, origin='lower', cmap='gray', interpolation='none')

        Background_array = None
        for idx, region in enumerate(background_line):
            limits_region = sorted(map(float, region.split(':')))
            low_limit, up_limit = limits_region[0], limits_region[1]
            trace_matrix = np.tile(trace_curve, (image_data.shape[1], 1)).T
            bg_mask = (x_grid > trace_matrix + low_limit) & (x_grid < trace_matrix + up_limit)
            masked_image = np.ma.masked_where(~bg_mask, image_data)
            if Background_array is None:
                Background_array = np.copy(masked_image)
            else:
                Background_array = np.hstack((Background_array, masked_image))

        background_flux = np.ma.sum(Background_array, axis=1)

        return background_flux

    def trace_to_frame(self, frame_address, axis_plot, trace_folder, ext=0, title_reference=''):

        original_file = frame_address[frame_address.rfind('/') + 1:frame_address.rfind('.fits')]
        trace_address = f'{trace_folder}/ap_{trace_folder[1:].replace("/", "_").replace("database", original_file)}'
        # trace_address = trace_folder + 'ap_' + trace_folder[1:].replace('/', '_').replace('_database_',
        #                                                                                   '_') + original_file.replace(
        #     '/', '_')
        # trace_address   = trace_address.replace('_standard_stars_','_objects_')

        # Read the text file
        file_trace = open(trace_address)
        file_lines = file_trace.readlines()
        file_trace.close()

        # Reference indeces
        idx_start = file_lines.index('\taxis\t1\n') + 1  # That '\t1' is the orientation of the frame
        idx_match = [i for i in range(len(file_lines)) if '\tcenter\t' in file_lines[i]]

        # Aperture data
        center_line = file_lines[idx_match[0]]
        lower_line = file_lines[idx_match[0] + 1]
        upper_line = file_lines[idx_match[0] + 2]
        # aper_center = map(float, center_line.split()[1:])
        # aper_low = map(float, lower_line.split()[1:])
        # aper_high = map(float, upper_line.split()[1:])
        aper_center = np.array(center_line.split()[1:]).astype(float)
        aper_low = np.array(lower_line.split()[1:]).astype(float)
        aper_high = np.array(upper_line.split()[1:]).astype(float)

        # Fitting coefficients
        coef_n = file_lines[idx_start].split()[1]
        fit_type = float(file_lines[idx_start + 1].split()[0])
        order = int(float(file_lines[idx_start + 2].split()[0]))
        xmin = int(float(file_lines[idx_start + 3].split()[0]))
        xmax = int(float(file_lines[idx_start + 4].split()[0]))
        coefs = np.empty(order)
        for i in range(len(coefs)):
            coefs[i] = float(file_lines[idx_start + 5 + i].split()[0])

            # Plot the polynomial
        y_range = np.arange(float(xmin), float(xmax))
        n = (2 * y_range - (xmax + xmin)) / (xmax - xmin)
        # poly_leg = np.legval(n, coefs)
        poly_leg = np.polynomial.legendre.legval(n, coefs)
        trace_curve = poly_leg + aper_center[0]
        low_limit = trace_curve + aper_low[0]
        high_limit = trace_curve + aper_high[0]
        axis_plot.plot(trace_curve, y_range, color='red', linestyle=':')
        axis_plot.fill_betweenx(y_range, low_limit, high_limit, alpha=0.3, facecolor='green', edgecolor='green',
                                linewidth=3.0)

        # Plot Background region
        idx_background = [i for i in range(len(file_lines)) if '\t\tsample' in file_lines[i]]
        background_line = file_lines[idx_background[0]][10:-2]
        background_intervals = background_line.split(',') if ',' in background_line else background_line.split(' ')

        for region in background_intervals:
            print(region)
            limits_region = np.array(region.split(':')).astype(float)
            low_limit_region = trace_curve + limits_region[0]
            high_limit_region = trace_curve + limits_region[1]
            axis_plot.fill_betweenx(y_range, low_limit_region, high_limit_region, alpha=0.2, facecolor='yellow')

        # Wording for the plot
        coefficients_line = r'$c_{{i}}$ = [{:s}]'.format(np.array_str(coefs, precision=3))
        # coefficients_line = r'$c_{{i}}$ = [{:s}]'.format(np.array_str(coefs, precision=3).translate(None, "["))
        title_lines = '{:s} {:s} (order {:d})\n({:s})\nApperture (pixels) {:.2f}'.format(title_reference, 'legendre',
                                                                                         order, coefficients_line,
                                                                                         aper_high[0] - aper_low[0])
        axis_plot.set_title(title_lines, fontsize=12)

        return

    def fits_compare(self, pdf_address, indeces_frame, ext=0, columns_mean=True):

        # Get data
        sorting_pattern = ['frame_tag', 'ISIARM']
        files_name = self.reducDf[indeces_frame].sort_values(sorting_pattern).file_name.values
        files_address = self.reducDf[indeces_frame].sort_values(sorting_pattern).file_location.values
        frames_color = self.reducDf[indeces_frame].sort_values(sorting_pattern).ISIARM.values
        frames_object = self.reducDf[indeces_frame].sort_values(sorting_pattern).OBJECT.values
        validity_check = self.reducDf[indeces_frame].sort_values(sorting_pattern).valid_file.values

        # Generate an artificial list with the range of files
        Number_frames = len(files_address)
        indices_list = range(Number_frames)

        # Figure configuration
        page_grid_width = 2
        self.Pdf_Fig, self.GridAxis = plt.subplots(1, page_grid_width, figsize=(10, 14))
        self.GridAxis_list = self.GridAxis.ravel()

        # Create the doc
        self.doc = Document(pdf_address)
        self.doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=0cm']))
        self.doc.append(NoEscape(r'\extrafloats{100}'))
        self.doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # Start looping through the indeces in groups with the same lengths as the columns * rows
        for sublist in grouper(page_grid_width, indices_list):
            for i in range(len(sublist)):

                # Get the index corresponding to the image to plot
                index_i = sublist[i]

                if index_i is not None:
                    colorTitle = 'black' if validity_check[index_i] else 'red'
                    fits_address = f'{files_address[index_i]}/{files_name[index_i]}'

                    self.fits_to_frame(fits_address, self.GridAxis_list[i],
                                       color=frames_color[index_i], ext=ext, valid=validity_check[index_i])
                    self.GridAxis_list[i].set_title(
                        '{filename}\n{object}'.format(filename=files_name[index_i], object=frames_object[index_i]),
                        fontsize=15, color=colorTitle)

            plt.tight_layout()

            # Add the plot
            with self.doc.create(Figure(position='htbp')) as plot:
                plot.add_plot()
                self.doc.append(NoEscape(r'\newpage'))
                plt.cla()

        # Plot mean row value
        if columns_mean:

            # Figure for the plot
            self.Pdf_Fig, self.GridAxis = plt.subplots(2, 1, figsize=(10, 12))  # figsize=(20, 24)
            self.GridAxis_list = self.GridAxis.ravel()

            # Loop through the files
            for j in range(len(files_name)):
                # Get label
                CodeName = files_name[j][0:files_name[j].find('.')]

                # Get axis according to the color
                axis_index = self.plots_dict[frames_color[j].replace('_arm', '')]

                # Plot the data
                file_location = f'{files_address[j]}/{files_name[j]}'
                self.plot_spectral_axis(file_location, self.GridAxis_list[axis_index], CodeName,
                                        ext=ext, valid=validity_check[j])

            # Plot layout
            for idx, val in enumerate(['Blue arm spectra', 'Red arm spectra']):
                self.GridAxis[idx].set_xlabel('Pixel value', fontsize=15)
                self.GridAxis[idx].set_ylabel('Mean spatial count', fontsize=15)
                self.GridAxis[idx].set_title(val, fontsize=20)
                self.GridAxis[idx].tick_params(axis='both', labelsize=14)
                self.GridAxis[idx].legend(loc='upper right', prop={'size': 14}, ncol=2)

            # Add the plot
            with self.doc.create(Figure(position='htbp')) as plot:
                plot.add_plot()
                self.doc.append(NoEscape(r'\newpage'))
                plt.cla()

        self.doc.generate_pdf(clean_tex=True)

        return

    def masked_pixels(self, pdf_address, indeces_frame, ext=0, columns_mean=True):

        fill = False

        # Get data
        sorting_pattern = ['RUN', 'frame_tag']
        files_name = self.reducDf[indeces_frame].sort_values(sorting_pattern).file_name.values
        files_address = self.reducDf[indeces_frame].sort_values(sorting_pattern).file_location.values
        frames_color = self.reducDf[indeces_frame].sort_values(sorting_pattern).ISIARM.values
        frames_object = self.reducDf[indeces_frame].sort_values(sorting_pattern).OBJECT.values
        validity_check = self.reducDf[indeces_frame].sort_values(sorting_pattern).valid_file.values

        # Generate an artificial list with the range of files
        Number_frames = len(files_address)
        indices_list = range(Number_frames)

        # Figure configuration
        page_grid_width = 2
        self.Pdf_Fig, self.GridAxis = plt.subplots(1, page_grid_width, figsize=(10, 14))
        self.GridAxis_list = self.GridAxis.ravel()

        # Create the doc
        self.doc = Document(pdf_address)
        self.doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=0cm']))
        self.doc.append(NoEscape(r'\extrafloats{100}'))
        self.doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # Start looping through the indeces in groups with the same lengths as the columns * rows
        for sublist in grouper(page_grid_width, indices_list):
            for i in range(len(sublist)):

                # Get the index corresponding to the image to plot
                index_i = sublist[i]

                if index_i is not None:
                    colorTitle = 'black' if validity_check[index_i] else 'red'
                    self.fits_to_frame(files_address[index_i] + files_name[index_i], self.GridAxis_list[i],
                                       color=frames_color[index_i], ext=ext, valid=validity_check[index_i])
                    self.GridAxis_list[i].set_title(
                        '{filename}\n{object}'.format(filename=files_name[index_i], object=frames_object[index_i]),
                        fontsize=15, color=colorTitle)

                    # load the mask
                    mask_file = '{rootfolder}badpix_{color}mask'.format(rootfolder=self.Catalogue_folder,
                                                                        color=frames_color[index_i].replace(' arm', ''))

                    c1, c2, r1, r2 = np.loadtxt(mask_file, dtype=int, comments='#', delimiter=' ', usecols=(0, 1, 2, 3),
                                             unpack=True, ndmin=2)

                    for j in range(len(c1)):
                        self.GridAxis_list[i].add_patch(
                            patches.Rectangle((c1[j], r1[j]), c2[j] - c1[j], r2[j] - r1[j], linewidth=0.5,
                                              color='yellow', fill=False))  # remove background

            plt.tight_layout()

            # Add the plot
            with self.doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(height=NoEscape(r'1\textheight'))
                self.doc.append(NoEscape(r'\newpage'))
                plt.cla()

        # Plot mean row value
        if columns_mean:

            # Figure for the plot
            self.Pdf_Fig, self.GridAxis = plt.subplots(2, 1, figsize=(10, 12))  # figsize=(20, 24)
            self.GridAxis_list = self.GridAxis.ravel()

            # Loop through the files
            for j in range(len(files_name)):
                # Get label
                CodeName = files_name[j][0:files_name[j].find('.')]

                # Get axis according to the color
                axis_index = self.plots_dict[frames_color[j].replace(' arm', '')]

                # Plot the data
                self.plot_spectral_axis(files_address[j] + files_name[j], self.GridAxis_list[axis_index], CodeName,
                                        ext=ext, valid=validity_check[j])

            # Plot layout
            for idx, val in enumerate(['Blue arm spectra', 'Red arm spectra']):
                self.GridAxis[idx].set_xlabel('Pixel value', fontsize=15)
                self.GridAxis[idx].set_ylabel('Mean spatial count', fontsize=15)
                self.GridAxis[idx].set_title(val, fontsize=20)
                self.GridAxis[idx].tick_params(axis='both', labelsize=14)
                self.GridAxis[idx].legend(loc='upper right', prop={'size': 14}, ncol=2)

            # Add the plot
            with self.doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(height=NoEscape(r'1\textheight'))
                self.doc.append(NoEscape(r'\newpage'))
                plt.cla()

        self.doc.generate_pdf(clean_tex=True)

        return

    def frame_combine(self, pdf_address, indx_frame_original, ext=0, sorting_pattern=['reduc_tag'], obs_conf=None):

        # Create the doc
        self.doc = Document(pdf_address)
        self.doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=1cm']))
        self.doc.append(NoEscape(r'\extrafloats{100}'))
        self.doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # Through through the objects to generate to create all the plots
        frames_objects = self.reducDf[indx_frame_original].frame_tag.unique()
        frames_colors = self.reducDf[indx_frame_original].ISIARM.unique()

        for frame_object in frames_objects:

            for arm_color in frames_colors:

                sub_indices = indx_frame_original & (self.reducDf.frame_tag == frame_object) & (
                            self.reducDf.ISIARM == arm_color)
                files_name = self.reducDf[sub_indices].sort_values(sorting_pattern, ascending=False).file_name.values
                files_address = self.reducDf[sub_indices].sort_values(sorting_pattern,
                                                                      ascending=False).file_location.values
                validity_check = self.reducDf[sub_indices].sort_values(sorting_pattern,
                                                                       ascending=False).valid_file.values

                Number_frames = len(files_address)
                indices_list = range(Number_frames)

                center_line_cords = obs_conf.get(f'{frame_object}_{arm_color}_ref_peak_array', None)

                # ---------------Figure complete frames
                page_grid_width = 3
                Pdf_Fig, GridAxis = plt.subplots(1, page_grid_width, figsize=(10, 14))
                GridAxis_list = GridAxis.ravel()

                # Start looping through the indeces in groups with the same lengths as the columns * rows
                for sublist in grouper(page_grid_width, indices_list):
                    for i in range(len(sublist)):

                        # Get the index corresponding to the image to plot
                        index_i = sublist[i]

                        if index_i is not None:
                            colorTitle = 'black' if validity_check[index_i] else 'red'
                            self.fits_to_frame(f'{files_address[index_i]}/{files_name[index_i]}', GridAxis_list[i],
                                               color=arm_color, ext=ext, valid=validity_check[index_i])
                            GridAxis_list[i].set_title(f'{files_name[index_i]}\n{frame_object}',
                                                        fontsize=15, color=colorTitle)

                        else:
                            GridAxis_list[i].clear()  # This seems abit too much but the tight layout crashes otherwise

                    plt.tight_layout()

                    # Add the plot
                    with self.doc.create(Figure(position='htbp')) as plot:
                        plot.add_plot()
                        self.doc.append(NoEscape(r'\newpage'))
                        plt.cla()

                # ---------------Figure reference lines maps
                grid_width, grid_height = 3, 2
                Pdf_Fig, GridAxis = plt.subplots(grid_height, grid_width, figsize=(10, 12))
                GridAxis_list = GridAxis.ravel()

                # Start looping through the indeces in groups with the same lengths as the columns * rows
                for sublist in grouper(grid_width * grid_height, indices_list):
                    for i in range(len(sublist)):

                        # Get the index corresponding to the image to plot
                        index_i = sublist[i]

                        if index_i is not None:
                            colorTitle = 'black' if validity_check[index_i] else 'red'
                            self.fits_to_frame(f'{files_address[index_i]}/{files_name[index_i]}', GridAxis_list[i],
                                               color=arm_color, ext=ext, valid=validity_check[index_i],
                                               section=center_line_cords)
                            GridAxis_list[i].set_title(f'{files_name[index_i]}\n{frame_object}',
                                                        fontsize=15, color=colorTitle)
                        else:
                            GridAxis_list[i].clear()  # This seems a bit too much but the tight layout crashes otherwise

                    # Add the plot
                    with self.doc.create(Figure(position='htbp')) as plot:
                        plot.add_plot()
                        self.doc.append(NoEscape(r'\newpage'))
                        plt.cla()

                # ---------------Reference line cut
                grid_width, grid_height = 1, 2
                Pdf_Fig, GridAxis = plt.subplots(grid_height, grid_width, figsize=(10, 12))

                for m in range(len(files_name)):

                    with fits.open(f'{files_address[m]}/{files_name[m]}') as hdu_list:
                        image_data = hdu_list[ext].data
                    print(files_name)
                    x_max, y_max = int(center_line_cords[0]), int(center_line_cords[1])
                    spatial_mean = image_data[y_max - 1:y_max + 1, :].mean(axis=0)
                    spatial_length = range(len(spatial_mean))

                    spectral_mean = image_data[:, x_max - 2:x_max + 2].mean(axis=1)
                    spectral_length = range(len(spectral_mean))

                    # Check if file is rejected:
                    if validity_check[m]:
                        label = files_name[m]
                    else:
                        label = 'REJECTED ' + files_name[m]

                    GridAxis[0].step(spatial_length, spatial_mean, label=label)
                    GridAxis[1].step(spectral_length, spectral_mean, label=label)

                # Plot layout
                GridAxis[0].set_xlabel('Pixel value', fontsize=15)
                GridAxis[0].set_ylabel('Mean spatial count', fontsize=15)
                GridAxis[0].set_title(f'{frame_object} {arm_color}\n{center_line_cords}', fontsize=15)
                GridAxis[0].tick_params(axis='both', labelsize=15)
                GridAxis[0].legend(loc='upper right', prop={'size': 12})
                GridAxis[0].set_xlim(x_max - 40, x_max + 40)

                GridAxis[1].set_xlabel('Pixel value', fontsize=15)
                GridAxis[1].set_ylabel('Mean spectral count', fontsize=15)
                GridAxis[1].set_title(f'{frame_object} {arm_color}\n{center_line_cords}', fontsize=15)
                GridAxis[1].tick_params(axis='both', labelsize=15)
                GridAxis[1].legend(loc='upper right', prop={'size': 12})
                GridAxis[1].set_yscale('log')

                # Add the plot
                with self.doc.create(Figure(position='htbp')) as plot:
                    plot.add_plot()
                    self.doc.append(NoEscape(r'\newpage'))
                    plt.cla()

        self.doc.generate_pdf(clean_tex=True)

    def frame_combine_shifted(self, pdf_address, indx_frame_original, ext=0, sorting_pattern=['reduc_tag']):

        # Create the doc
        self.doc = Document(pdf_address)
        self.doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=1cm']))
        self.doc.append(NoEscape(r'\extrafloats{100}'))
        self.doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # Through through the objects to generate to create all the plots
        frames_objects = self.reducDf[indx_frame_original].frame_tag.unique()
        frames_colors = self.reducDf[indx_frame_original].ISIARM.unique()

        for frame_object in frames_objects:

            for arm_color in frames_colors:

                sub_indices = indx_frame_original & (self.reducDf.frame_tag == frame_object) & (
                            self.reducDf.ISIARM == arm_color)
                files_name = self.reducDf[sub_indices].sort_values(sorting_pattern, ascending=False).file_name.values
                files_address = self.reducDf[sub_indices].sort_values(sorting_pattern,
                                                                      ascending=False).file_location.values
                validity_check = self.reducDf[sub_indices].sort_values(sorting_pattern,
                                                                       ascending=False).valid_file.values

                Number_frames = len(files_address)
                indices_list = range(Number_frames)
                center_line_cords = self.observation_dict[
                    '{CodeName}_refline_{Color}'.format(CodeName=frame_object, Color=arm_color.replace(' arm', ''))]

                # ---------------Figure complete frames
                page_grid_width = 3
                Pdf_Fig, GridAxis = plt.subplots(1, page_grid_width, figsize=(10, 14))
                GridAxis_list = GridAxis.ravel()

                # Start looping through the indeces in groups with the same lengths as the columns * rows
                for sublist in grouper(page_grid_width, indices_list):
                    for i in range(len(sublist)):

                        # Get the index corresponding to the image to plot
                        index_i = sublist[i]

                        if index_i is not None:
                            colorTitle = 'black' if validity_check[index_i] else 'red'
                            self.fits_to_frame(files_address[index_i] + files_name[index_i], GridAxis_list[i],
                                               color=arm_color, ext=ext, valid=validity_check[index_i])
                            GridAxis_list[i].set_title(
                                '{filename}\n{object}'.format(filename=files_name[index_i], object=frame_object),
                                fontsize=15, color=colorTitle)

                        else:
                            GridAxis_list[i].clear()  # This seems abit too much but the tight layout crashes otherwise

                    plt.tight_layout()

                    # Add the plot
                    with self.doc.create(Figure(position='htbp')) as plot:
                        plot.add_plot()
                        self.doc.append(NoEscape(r'\newpage'))
                        plt.cla()

                # ---------------Figure reference lines maps
                grid_width, grid_height = 3, 2
                Pdf_Fig, GridAxis = plt.subplots(grid_height, grid_width, figsize=(10, 12))
                GridAxis_list = GridAxis.ravel()

                # Start looping through the indeces in groups with the same lengths as the columns * rows
                for sublist in grouper(grid_width * grid_height, indices_list):
                    for i in range(len(sublist)):

                        # Get the index corresponding to the image to plot
                        index_i = sublist[i]

                        if index_i is not None:
                            colorTitle = 'black' if validity_check[index_i] else 'red'
                            self.fits_to_frame(files_address[index_i] + files_name[index_i], GridAxis_list[i],
                                               color=arm_color, ext=ext, valid=validity_check[index_i],
                                               section=center_line_cords)
                            GridAxis_list[i].set_title(
                                '{filename}\n{object}'.format(filename=files_name[index_i], object=frame_object),
                                fontsize=15, color=colorTitle)
                        else:
                            GridAxis_list[i].clear()  # This seems a bit too much but the tight layout crashes otherwise

                    # Add the plot
                    with self.doc.create(Figure(position='htbp')) as plot:
                        plot.add_plot()
                        self.doc.append(NoEscape(r'\newpage'))
                        plt.cla()

                # ---------------Reference line cut
                grid_width, grid_height = 1, 2
                Pdf_Fig, GridAxis = plt.subplots(grid_height, grid_width, figsize=(10, 12))

                for m in range(len(files_name)):

                    with fits.open(files_address[m] + files_name[m]) as hdu_list:
                        image_data = hdu_list[ext].data

                    x_max, y_max = int(center_line_cords[0]), int(center_line_cords[1])
                    spatial_mean = image_data[y_max - 1:y_max + 1, :].mean(axis=0)
                    spatial_length = range(len(spatial_mean))

                    spectral_mean = image_data[:, x_max - 2:x_max + 2].mean(axis=1)
                    spectral_length = range(len(spectral_mean))

                    # Check if file is rejected:
                    if validity_check[m]:
                        label = files_name[m]
                    else:
                        label = 'REJECTED ' + files_name[m]

                    GridAxis[0].step(spatial_length, spatial_mean, label=label)
                    GridAxis[1].step(spectral_length, spectral_mean, label=label)

                # Plot layout
                GridAxis[0].set_xlabel('Pixel value', fontsize=15)
                GridAxis[0].set_ylabel('Mean spatial count', fontsize=15)
                GridAxis[0].set_title(frame_object + ' ' + arm_color, fontsize=15)
                GridAxis[0].tick_params(axis='both', labelsize=15)
                GridAxis[0].legend(loc='upper right', prop={'size': 12})
                GridAxis[0].set_xlim(x_max - 40, x_max + 40)

                GridAxis[1].set_xlabel('Pixel value', fontsize=15)
                GridAxis[1].set_ylabel('Mean spectral count', fontsize=15)
                GridAxis[1].set_title(frame_object + ' ' + arm_color, fontsize=15)
                GridAxis[1].tick_params(axis='both', labelsize=15)
                GridAxis[1].legend(loc='upper right', prop={'size': 12})
                GridAxis[1].set_yscale('log')

                # Add the plot
                with self.doc.create(Figure(position='htbp')) as plot:
                    plot.add_plot()
                    self.doc.append(NoEscape(r'\newpage'))
                    plt.cla()

        self.doc.generate_pdf(clean_tex=True)

    def objects_focus(self, pdf_address, indx_frame_original, ext=0, sorting_pattern=['ISIARM']):

        # Create the doc
        self.doc = Document(pdf_address)
        self.doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=1cm']))
        self.doc.append(NoEscape(r'\extrafloats{100}'))
        self.doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # Through through the objects to generate to create all the plots
        frames_objects = self.reducDf[indx_frame_original].frame_tag.unique()
        frames_colors = self.reducDf[indx_frame_original].ISIARM.unique()

        for frame_object in frames_objects:

            for j in range(len(frames_colors)):

                grid_width, grid_height = 1, 2
                Pdf_Fig, GridAxis = plt.subplots(grid_height, grid_width, figsize=(10, 12))

                sub_indices = indx_frame_original & (self.reducDf.frame_tag == frame_object) & (
                            self.reducDf.ISIARM == frames_colors[j])
                files_name = self.reducDf[sub_indices].sort_values(sorting_pattern, ascending=False).file_name.values
                files_address = self.reducDf[sub_indices].sort_values(sorting_pattern,
                                                                      ascending=False).file_location.values
                validity_check = self.reducDf[sub_indices].sort_values(sorting_pattern,
                                                                       ascending=False).valid_file.values

                for perspective in [0, 1]:

                    for m in range(len(files_name)):

                        with fits.open(files_address[m] + files_name[m]) as hdu_list:
                            image_data = hdu_list[ext].data

                        # Check if file is rejected:
                        if validity_check[m]:
                            label = files_name[m]
                        else:
                            label = 'REJECTED ' + files_name[m]

                        if perspective == 0:
                            # x_max, y_max       = int(center_line_cords[0]), int(center_line_cords[1])
                            # spatial_mean       = image_data[y_max-1:y_max+1, :].mean(axis=0)
                            spatial_mean = image_data.mean(axis=0)
                            spatial_length = range(len(spatial_mean))
                            GridAxis[perspective].step(spatial_length, spatial_mean, label=label)

                        else:
                            # spectral_mean       = image_data[:, x_max-2:x_max+2].mean(axis=1)
                            spectral_mean = image_data.mean(axis=1)
                            spectral_length = range(len(spectral_mean))
                            GridAxis[perspective].step(spectral_length, spectral_mean, label=label)

                    # Plot layout
                    GridAxis[perspective].set_xlabel('Pixel', fontsize=15)
                    GridAxis[perspective].set_ylabel('Mean spatial count', fontsize=15)
                    GridAxis[perspective].set_title(frame_object + ' ' + frames_colors[j], fontsize=15)
                    GridAxis[perspective].tick_params(axis='both', labelsize=15)
                    GridAxis[perspective].legend(loc='upper right', prop={'size': 12})

                plt.tight_layout()

                # Add the plot
                with self.doc.create(Figure(position='htbp')) as plot:
                    plot.add_plot(height=NoEscape(r'1\textheight'))
                    self.doc.append(NoEscape(r'\newpage'))
                    plt.cla()

        self.doc.generate_pdf(clean_tex=True)

    def cosmic_plot(self, pdf_address, indeces_frame, ext=0, colors_dict=None, columns_mean=True):

        # Get data
        sorting_pattern = ['ISIARM', 'frame_tag']
        files_name = self.reducDf[indeces_frame].sort_values(sorting_pattern).file_name.values
        files_address = self.reducDf[indeces_frame].sort_values(sorting_pattern).file_location.values
        frames_color = self.reducDf[indeces_frame].sort_values(sorting_pattern).ISIARM.values
        frames_object = self.reducDf[indeces_frame].sort_values(sorting_pattern).OBJECT.values

        # Generate an artificial list with the range of files
        Number_frames = len(files_address)

        # Figure configuration
        titles = ['Original', 'Mask', 'cleaned']
        page_grid_width = 3
        self.Pdf_Fig, self.GridAxis = plt.subplots(1, page_grid_width, figsize=(10, 14))
        self.GridAxis_list = self.GridAxis.ravel()

        # Create the doc
        self.doc = Document(pdf_address)
        self.doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=0cm']))
        self.doc.append(NoEscape(r'\extrafloats{100}'))
        self.doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # Start looping through the indeces in groups with the same lengths as the columns * rows
        for i in range(Number_frames):

            clean_name = f'{files_address[i]}/{files_name[i]}'
            input_name = clean_name.replace('_cr.fits', '.fits')
            mask_name = clean_name.replace('_cr.fits', '_mask.fits')
            plot_frames = [input_name, input_name, clean_name]

            for j in range(3):

                # Plot the frame
                self.fits_to_frame(plot_frames[j], self.GridAxis_list[j], color=frames_color[i], ext=ext)
                self.GridAxis_list[j].set_title(
                    '{filename}\n{object}\n{type_f}'.format(filename=files_name[i], object=frames_object[i],
                                                            type_f=titles[j]), fontsize=15)

                # Overplot cosmic rays location
                if j == 1:
                    with fits.open(mask_name) as hdu_list:
                        image_mask = hdu_list[ext].data

                    mask_pixels = np.where(image_mask == 1)
                    xValues, yValues = mask_pixels[1], mask_pixels[0]

                    self.GridAxis_list[j].scatter(xValues, yValues, s=15, edgecolor='yellow', facecolor='none')

            plt.tight_layout()

            # Add the plot
            with self.doc.create(Figure(position='htbp')) as plot:
                plot.add_plot()
                self.doc.append(NoEscape(r'\newpage'))
                plt.cla()

        self.doc.generate_pdf(clean_tex=True)

    def cosmic_plot2(self, pdf_address, indeces_frame, ext=0, colors_dict=None, columns_mean=True):

        files_name = self.reducDf[indeces_frame].sort_values(['frame_tag']).file_name.values
        files_address = self.reducDf[indeces_frame].sort_values(['frame_tag']).file_location.values
        frames_color = self.reducDf[indeces_frame].sort_values(['frame_tag']).ISIARM.values
        slit_widths = self.reducDf[indeces_frame].sort_values(['frame_tag']).ISISLITW.values
        tags_frame = self.reducDf[indeces_frame].sort_values(['frame_tag']).frame_tag.values
        titles = ['Original', 'cleaned', 'masked']
        Number_frames = len(files_address)

        # Figure configuration
        Pdf_Fig, GridAxis = plt.subplots(1, self.page_grid_width, figsize=(20, 24))
        GridAxis_list = GridAxis.ravel()

        # Create the doc
        doc = Document(pdf_address)
        doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=1cm']))
        doc.append(NoEscape(r'\extrafloats{400}'))
        doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # Start looping through the indeces in groups with the same lengths as the columns * rows
        for i in range(Number_frames):

            clean_name = files_address[i] + files_name[i]
            input_name = clean_name.replace('_cr.fits', '.fits')
            mask_name = clean_name.replace('_cr.fits', '_mask.fits')
            plot_frames = [input_name, input_name, clean_name]

            for j in range(3):

                # Clear the axis from previous plot
                GridAxis_list[j].clear()
                GridAxis_list[j].axes.get_yaxis().set_visible(False)
                GridAxis_list[j].axes.get_xaxis().set_visible(False)

                with fits.open(plot_frames[j]) as hdu_list:
                    image_data = hdu_list[ext].data

                # Get zscale limits for plotting the image
                IntensityLimits = ZScaleInterval()
                int_min, int_max = IntensityLimits.get_limits(image_data)[0], IntensityLimits.get_limits(image_data)[1]

                # Plot the data
                cmap_frame = colors_dict[frames_color[i]]
                GridAxis_list[j].imshow(image_data, cmap=cmap_frame, origin='lower', vmin=int_min, vmax=int_max,
                                        interpolation='nearest', aspect='auto')
                GridAxis_list[j].set_title(
                    files_name[i] + '\n' + titles[j] + '\n' + tags_frame[i] + ' ' + str(slit_widths[i]), fontsize=12)
                plt.axis('tight')

                # Overplot cosmic rays location
                if j == 1:
                    with fits.open(mask_name) as hdu_list:
                        image_mask = hdu_list[ext].data

                    mask_pixels = np.where(image_mask == 1)
                    xValues = mask_pixels[1]
                    yValues = mask_pixels[0]

                    GridAxis_list[j].scatter(xValues, yValues, s=15, edgecolor='yellow', facecolor='none')
                    plt.axis('tight')
                plt.axis('tight')

            # Add the plot
            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(height=NoEscape(r'1\textheight'))
                doc.append(NoEscape(r'\newpage'))
                plt.cla()

        if columns_mean:

            # Figure for the plot
            Pdf_Fig, GridAxis = plt.subplots(2, 1, figsize=(20, 24))
            GridAxis_list = GridAxis.ravel()

            linestyles = ["-", "-", "--", "--", "-.", "-.", ":", ":"]
            linecycler = cycle(linestyles)

            for j in range(len(files_name)):

                CodeName = files_name[j][0:files_name[j].find('.')]

                with fits.open(files_address[j] + files_name[j]) as hdu_list:
                    image_data = hdu_list[ext].data

                y_values = image_data.mean(axis=1)
                x_values = range(len(y_values))

                if frames_color[j] == 'Blue arm':
                    GridAxis_list[0].plot(x_values, y_values, label=CodeName, linestyle=next(linecycler))
                else:
                    GridAxis_list[1].plot(x_values, y_values, label=CodeName, linestyle=next(linecycler))

            # Plot layout
            plt.axis('tight')
            for idx, val in enumerate(['Blue arm spectra', 'Red arm spectra']):
                GridAxis[idx].set_xlabel('Pixel value', fontsize=30)
                GridAxis[idx].set_ylabel('Mean spatial count', fontsize=30)
                GridAxis[idx].set_title(val, fontsize=30)
                GridAxis[idx].tick_params(axis='both', labelsize=20)
                GridAxis[idx].legend(loc='upper right', prop={'size': 12})

            # Add the plot
            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(height=NoEscape(r'1\textheight'))
                doc.append(NoEscape(r'\newpage'))
                plt.cla()

        doc.generate_pdf(clean_tex=True)

    def extracted_frames(self, pdf_address, indeces_frame, ext=0):

        files_name = self.reducDf[indeces_frame].sort_values(['frame_tag', 'ISIARM']).file_name.values
        files_address = self.reducDf[indeces_frame].sort_values(['frame_tag', 'ISIARM']).file_location.values
        frames_color = self.reducDf[indeces_frame].sort_values(['frame_tag', 'ISIARM']).ISIARM.values
        frames_object = self.reducDf[indeces_frame].sort_values(['frame_tag', 'ISIARM']).OBJECT.values

        # Generate an artificial list with the range of files
        Number_frames = len(files_address)
        indices_list = range(Number_frames)

        # Figure configuration
        page_grid_width = 2
        self.Pdf_Fig, self.GridAxis = plt.subplots(1, page_grid_width, figsize=(8, 12))
        self.GridAxis_list = self.GridAxis.ravel()

        # Create the doc
        self.doc = Document(pdf_address)
        self.doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=1cm']))
        self.doc.append(NoEscape(r'\extrafloats{100}'))
        self.doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # ------Plotting trace curve observations
        for sublist in grouper(page_grid_width, indices_list):
            for i in range(len(sublist)):

                # Get the index corresponding to the image to plot
                index_i = sublist[i]
                if index_i is not None:
                    # Plot the fits frame
                    file_name_2d = files_name[index_i].replace('_e.fits', '.fits')
                    self.fits_to_frame(f'{files_address[index_i]}/{file_name_2d}', self.GridAxis_list[i],
                                       color=frames_color[index_i])

                    # Plot the trace
                    self.trace_to_frame(f'{files_address[index_i]}/{file_name_2d}', self.GridAxis_list[i],
                                        f'{files_address[index_i]}/database',
                                        title_reference='{:s}\n'.format(files_name[index_i]))

            plt.tight_layout()

            # Add the plot
            with self.doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(height=NoEscape(r'1\textheight'))
                self.doc.append(NoEscape(r'\newpage'))
                plt.cla()

        # ------Plotting extracted observations
        self.Pdf_Fig = plt.figure(figsize=(16, 10))
        self.GridAxis = self.Pdf_Fig.add_subplot(111)

        # Loop through the fits data:
        for i in range(Number_frames):

            wavelength, Flux_array, Header_0 = self.get_spectra_data(f'{files_address[i]}/{files_name[i]}')

            if ('NAXIS2' in Header_0) and ('NAXIS3' in Header_0):
                appertures_number = Header_0['NAXIS2']
                spectra_number = 1
                for apper_n in range(appertures_number):
                    self.GridAxis.plot(wavelength, Flux_array[0][apper_n],
                                       label='extracted spectrum apperture: {apper_n}'.format(apper_n=apper_n + 1))

            elif ('NAXIS2' in Header_0) and ('NAXIS1' in Header_0):
                appertures_number = Header_0['NAXIS2']
                spectra_number = 1
                for apper_n in range(appertures_number):
                    self.GridAxis.plot(wavelength, Flux_array[apper_n],
                                       label='extracted spectrum apperture: {apper_n}'.format(apper_n=apper_n + 1))

            else:
                spectra_number = 1
                appertures_number = 1
                self.GridAxis.plot(wavelength, Flux_array, label='extracted spectrum')

            # Plot wording
            self.GridAxis.set_xlabel(r'Wavelength $(\AA)$', fontsize=20)
            self.GridAxis.set_ylabel('Flux' + r'$(erg\,cm^{-2} s^{-1} \AA^{-1})$', fontsize=20)
            self.GridAxis.set_title(files_name[i] + ' extracted spectrum', fontsize=25)
            self.GridAxis.legend()

            # Add the plot
            with self.doc.create(Figure(position='htbp')) as plot:
                plot.add_plot()
                self.doc.append(NoEscape(r'\newpage'))
                plt.cla()

        # Generate pdf
        self.doc.generate_pdf(clean_tex=True)

        return

    def flats_plotting(self, pdf_address, indeces_frame, ext=0, columns_mean=False):

        files_name = self.reducDf[indeces_frame].sort_values(['frame_tag', 'ISIARM']).file_name.values
        files_address = self.reducDf[indeces_frame].sort_values(['frame_tag', 'ISIARM']).file_location.values
        frames_color = self.reducDf[indeces_frame].sort_values(['frame_tag', 'ISIARM']).ISIARM.values
        frames_object = self.reducDf[indeces_frame].sort_values(['frame_tag', 'ISIARM']).OBJECT.values

        # Generate an artificial list with the range of files
        Number_frames = len(files_address)
        indices_list = range(Number_frames)

        # Figure configuration
        page_grid_width = 4
        self.Pdf_Fig, self.GridAxis = plt.subplots(1, page_grid_width, figsize=(30, 40))
        self.GridAxis_list = self.GridAxis.ravel()

        # Create the doc
        self.doc = Document(pdf_address)
        self.doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=1cm']))
        self.doc.append(NoEscape(r'\extrafloats{100}'))
        self.doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # Start looping through the indeces in groups with the same lengths as the columns * rows
        for sublist in grouper(page_grid_width, indices_list):
            for i in range(len(sublist)):

                # Get the index corresponding to the image to plot
                index_i = sublist[i]

                if index_i is not None:
                    self.fits_to_frame(files_address[index_i] + files_name[index_i], self.GridAxis_list[i],
                                       frames_color[index_i], ext=ext)
                    self.GridAxis_list[i].set_title(
                        '{filename}\n{object}'.format(filename=files_name[index_i], object=frames_object[index_i]),
                        fontsize=25)

                    # plot spectral limits for matching high an low orders:
                if 'Flat_limits' in self.observation_dict:
                    x_limits = self.GridAxis_list[i].get_xlim()
                    limits_flat = map(int, self.observation_dict['Flat_limits'])
                    if 'Blue' in frames_color[index_i]:
                        self.GridAxis_list[i].plot(x_limits, (limits_flat[0], limits_flat[0]), color='purple',
                                                   linewidth=3, linestyle='-.')
                        self.GridAxis_list[i].plot(x_limits, (limits_flat[1], limits_flat[1]), color='purple',
                                                   linewidth=3, linestyle='-.')
                    else:
                        self.GridAxis_list[i].plot(x_limits, (limits_flat[2], limits_flat[2]), color='purple',
                                                   linewidth=3, linestyle='-.')
                        self.GridAxis_list[i].plot(x_limits, (limits_flat[3], limits_flat[3]), color='purple',
                                                   linewidth=3, linestyle='-.')

                        # Add the plot
            with self.doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(height=NoEscape(r'1\textheight'))
                self.doc.append(NoEscape(r'\newpage'))
                plt.cla()

        # Plotting spectral axis conf with narrow limits
        if columns_mean:

            # Figure configuration
            self.Pdf_Fig, self.GridAxis = plt.subplots(2, 1, figsize=(10, 12))  # figsize=(20, 24)
            self.GridAxis_list = self.GridAxis.ravel()

            # Loop through files
            for j in range(len(files_name)):
                # Get label
                CodeName = files_name[j][0:files_name[j].find('.')]

                # Get axis according to the color
                axis_index = self.plots_dict[frames_color[j].replace(' arm', '')]

                # Plot the data
                self.plot_spectral_axis(files_address[j] + files_name[j], self.GridAxis_list[axis_index], CodeName)

            # plot spectral limits for matching high an low orders:
            if 'Flat_limits' in self.observation_dict:
                limits_flat = map(int, self.observation_dict['Flat_limits'])
                self.GridAxis[0].axvline(limits_flat[0], color='red', linewidth=1, linestyle='--')
                self.GridAxis[0].axvline(limits_flat[1], color='red', linewidth=2, linestyle='--')
                self.GridAxis[1].axvline(limits_flat[2], color='red', linewidth=1, linestyle='--')
                self.GridAxis[1].axvline(limits_flat[3], color='red', linewidth=2, linestyle='--')

                # Plot cropping limits
            for idx, val in enumerate(['Blue', 'Red']):
                cropping = map(int, self.observation_dict[val + '_cropping'])
                self.GridAxis[idx].axvline(cropping[2], color='red', linewidth=1)
                self.GridAxis[idx].axvline(cropping[3], color='red', linewidth=2)
                self.GridAxis[idx].set_xlim(cropping[2] - 10, cropping[3] + 10)
                self.GridAxis[idx].set_ylim(0.3, 1.3)

            # Plot layout
            for idx, val in enumerate(['Blue arm spectra', 'Red arm spectra']):
                self.GridAxis[idx].set_xlabel('Pixel value', fontsize=15)
                self.GridAxis[idx].set_ylabel('Mean spatial count', fontsize=15)
                self.GridAxis[idx].set_title(val, fontsize=20)
                self.GridAxis[idx].tick_params(axis='both', labelsize=14)
                self.GridAxis[idx].legend(loc='upper right', prop={'size': 14}, ncol=2)

            # Add the plot
            with self.doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(height=NoEscape(r'1\textheight'))
                self.doc.append(NoEscape(r'\newpage'))
                plt.cla()

                # Plotting spectral axis conf with narrow limits
        if columns_mean:

            # Figure configuration
            self.Pdf_Fig, self.GridAxis = plt.subplots(2, 1, figsize=(10, 12))  # figsize=(20, 24)
            self.GridAxis_list = self.GridAxis.ravel()

            # Loop through files
            for j in range(len(files_name)):
                # Get label
                CodeName = files_name[j][0:files_name[j].find('.')]

                # Get axis according to the color
                axis_index = self.plots_dict[frames_color[j].replace(' arm', '')]

                # Plot the data
                self.plot_spectral_axis(files_address[j] + files_name[j], self.GridAxis_list[axis_index], CodeName)

            # plot spectral limits for matching high an low orders:
            if 'Flat_limits' in self.observation_dict:
                limits_flat = map(int, self.observation_dict['Flat_limits'])
                self.GridAxis[0].axvline(limits_flat[0], color='purple', linewidth=1)
                self.GridAxis[0].axvline(limits_flat[1], color='purple', linewidth=2)
                self.GridAxis[1].axvline(limits_flat[2], color='purple', linewidth=1)
                self.GridAxis[1].axvline(limits_flat[3], color='purple', linewidth=2)

                # Plot cropping limits
            for idx, val in enumerate(['Blue', 'Red']):
                cropping = map(int, self.observation_dict[val + '_cropping'])
                self.GridAxis[idx].axvline(cropping[2], color='red', linewidth=1)
                self.GridAxis[idx].axvline(cropping[3], color='red', linewidth=2)

            # Plot layout
            for idx, val in enumerate(['Blue arm spectra', 'Red arm spectra']):
                self.GridAxis[idx].set_xlabel('Pixel value', fontsize=15)
                self.GridAxis[idx].set_ylabel('Mean spatial count', fontsize=15)
                self.GridAxis[idx].set_title(val, fontsize=20)
                self.GridAxis[idx].tick_params(axis='both', labelsize=14)
                self.GridAxis[idx].legend(loc='upper right', prop={'size': 14}, ncol=2)

            # Add the plot
            with self.doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(height=NoEscape(r'1\textheight'))
                self.doc.append(NoEscape(r'\newpage'))
                plt.cla()

        self.doc.generate_pdf(clean_tex=True)

    def fits_catalogue(self, pdf_address, indeces_frame, ext=0, columns_mean=False,
                       sorting_pattern=['frame_tag', 'ISIARM']):

        files_name = self.reducDf[indeces_frame].sort_values(sorting_pattern).file_name.values
        files_address = self.reducDf[indeces_frame].sort_values(sorting_pattern).file_location.values
        frames_color = self.reducDf[indeces_frame].sort_values(sorting_pattern).ISIARM.values
        frames_object = self.reducDf[indeces_frame].sort_values(sorting_pattern).OBJECT.values
        validity_check = self.reducDf[indeces_frame].sort_values(sorting_pattern).valid_file.values

        # Generate an artificial list with the range of files
        Number_frames = len(files_address)
        indices_list = range(Number_frames)

        # Figure configuration
        page_grid_width = 5
        self.Pdf_Fig, self.GridAxis = plt.subplots(1, page_grid_width, figsize=(10, 14))
        self.GridAxis_list = self.GridAxis.ravel()

        # Create the doc
        self.doc = Document(pdf_address)
        self.doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=0cm']))
        self.doc.append(NoEscape(r'\extrafloats{100}'))
        self.doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # Start looping through the indeces in groups with the same lengths as the columns * rows
        for sublist in grouper(page_grid_width, indices_list):
            for i in range(len(sublist)):

                # Get the index corresponding to the image to plot
                index_i = sublist[i]

                if index_i is not None:

                    fits_address = f'{files_address[index_i]}/{files_name[index_i]}'
                    self.fits_to_frame(fits_address, self.GridAxis_list[i], frames_color[index_i], ext=ext,
                                       valid=validity_check[index_i])

                    colorTitle = 'black' if validity_check[index_i] else 'red'
                    fits_title = f'{files_name[index_i]}\n{frames_object[index_i]}'
                    self.GridAxis_list[i].set_title(fits_title, fontsize=12, color=colorTitle)

            plt.tight_layout()

            # Add the plot
            with self.doc.create(Figure(position='htbp')) as plot:
                plot.add_plot()
                self.doc.append(NoEscape(r'\newpage'))
                plt.cla()

            for axis in self.GridAxis_list:
                axis.clear()

        if columns_mean:

            # Figure for the plot
            self.Pdf_Fig, self.GridAxis = plt.subplots(2, 1, figsize=(10, 12))  # figsize=(20, 24)
            self.GridAxis_list = self.GridAxis.ravel()

            # Loop through the files
            for j in range(len(files_name)):

                # Get label
                CodeName = files_name[j][0:files_name[j].find('.')]

                # Get axis according to the color
                if frames_color[j] in self.frames_colors:
                    axis_index = self.plots_dict[frames_color[j].replace('_arm', '')]
                else:
                    axis_index = 0

                # Plot the data
                fits_address = f'{files_address[j]}/{files_name[j]}'
                self.plot_spectral_axis(fits_address, self.GridAxis_list[axis_index], CodeName,
                                        ext=ext, valid=validity_check[j])

            # Plot layout
            for idx, val in enumerate(['Blue arm spectra', 'Red arm spectra']):
                self.GridAxis[idx].set_xlabel('Pixel value', fontsize=15)
                self.GridAxis[idx].set_ylabel('Mean spatial count', fontsize=15)
                self.GridAxis[idx].set_title(val, fontsize=20)
                self.GridAxis[idx].tick_params(axis='both', labelsize=14)
                self.GridAxis[idx].legend(loc='upper right', prop={'size': 14}, ncol=2)

            # Add the plot
            with self.doc.create(Figure(position='htbp')) as plot:
                plot.add_plot()
                # plot.add_plot(height=NoEscape(r'1\textheight'))
                self.doc.append(NoEscape(r'\newpage'))
                plt.cla()

        self.doc.generate_pdf(clean_tex=True)

    def arcs_compare(self, pdf_address, files_address, files_name, frames_color, ext=0, colors_dict=None):

        # Generate an artificial list with the range of files
        Number_frames = len(files_address)

        # Create the doc
        doc = Document(pdf_address)
        doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=1cm']))
        doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # Start looping through the indeces in groups with the same lengths as the columns * rows
        for i in range(Number_frames):

            # Load the image #need to remove this check.. simply add the files to rejected images
            hdu_list = fits.open(files_address[i] + files_name[i])
            image_head = hdu_list[ext].header
            image_data = hdu_list[ext].data  # This might be different for non frames
            wcs = WCS(image_head)
            hdu_list.close()

            # Figure configuration
            # Pdf_Fig, GridAxis   = plt.subplots(figsize=(20, 24), projection=wcs)
            Pdf_Fig, GridAxis = plt.subplots(figsize=(20, 28), ncols=2, subplot_kw={'projection': wcs})
            GridAxis_list = GridAxis.ravel()

            # Get zscale limits for plotting the image
            IntensityLimits = ZScaleInterval()
            int_min, int_max = IntensityLimits.get_limits(image_data)[0], IntensityLimits.get_limits(image_data)[1]

            # Plot the data
            cmap_frame = colors_dict[frames_color[i]]
            GridAxis_list[0].imshow(image_data, cmap=cmap_frame, origin='lower', vmin=int_min, vmax=int_max,
                                    interpolation='nearest', aspect='auto')
            GridAxis_list[1].imshow(image_data, cmap=cmap_frame, origin='lower', vmin=int_min, vmax=int_max,
                                    interpolation='nearest', aspect='auto')

            GridAxis_list[0].set_title('Combined arc', fontsize=28)
            GridAxis_list[1].set_title('Combined arc\nwith reference wavelengths', fontsize=28)

            # Plot the lines
            x_limits = GridAxis_list[1].get_xlim()
            if frames_color[i] == 'Blue arm':
                for line_wavelength in self.Blue_Arc_lines:
                    GridAxis_list[1].plot(x_limits, (line_wavelength, line_wavelength), color='purple', linewidth=3,
                                          linestyle='-.', transform=GridAxis_list[1].get_transform('world'))
            else:
                for line_wavelength in self.Red_Arc_lines:
                    GridAxis_list[1].plot(x_limits, (line_wavelength, line_wavelength), color='purple', linewidth=3,
                                          linestyle='-.', transform=GridAxis_list[1].get_transform('world'))

            plt.axis('tight')

            # Add the plot
            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(height=NoEscape(r'1\textheight'))
                doc.append(NoEscape(r'\newpage'))
                plt.cla()

        doc.generate_pdf(clean_tex=True)

    def fast_combined(self, pdf_address, indeces_frame, observation_dict, reducDf, ext=0, colors_dict=None):

        files_name = self.reducDf[indeces_frame].sort_values(['frame_tag']).file_name.values
        files_address = self.reducDf[indeces_frame].sort_values(['frame_tag']).file_location.values
        frames_color = self.reducDf[indeces_frame].sort_values(['frame_tag']).ISIARM.values
        frames_object = self.reducDf[indeces_frame].sort_values(['frame_tag']).frame_tag.values

        # Generate an artificial list with the range of files
        Number_frames = len(files_address)
        indices_list = range(Number_frames)

        # Figure configuration
        Pdf_Fig, GridAxis = plt.subplots(1, 2, figsize=(20, 24))
        GridAxis_list = GridAxis.ravel()

        # Create the doc
        doc = Document(pdf_address)
        doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=1cm']))
        doc.append(NoEscape(r'\extrafloats{100}'))
        doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # Start looping through the indeces in groups with the same lengths as the columns * rows
        for sublist in grouper(2, indices_list):
            for i in range(len(sublist)):

                # Get the index corresponding to the image to plot
                index_i = sublist[i]

                # Clear the axis from previous plot
                GridAxis_list[i].clear()
                GridAxis_list[i].axes.get_yaxis().set_visible(False)
                GridAxis_list[i].axes.get_xaxis().set_visible(False)

                if index_i is not None:
                    with fits.open(files_address[index_i] + files_name[index_i]) as hdu_list:
                        image_data = hdu_list[ext].data

                    # Get zscale limits for plotting the image
                    IntensityLimits = ZScaleInterval()
                    int_min, int_max = IntensityLimits.get_limits(image_data)[0], \
                                       IntensityLimits.get_limits(image_data)[1]

                    # Plot the data
                    GridAxis_list[i].clear()
                    GridAxis_list[i].imshow(image_data, cmap=colors_dict[frames_color[index_i]], origin='lower',
                                            vmin=int_min, vmax=int_max, interpolation='nearest', aspect='auto')
                    GridAxis_list[i].set_title(files_name[index_i], fontsize=20)

                    # Loading the reference line location
                    arm_color = frames_color[index_i].split()[0]
                    obj_refLine_key = '{CodeName}_refline_{Color}'.format(CodeName=frames_object[index_i],
                                                                          Color=arm_color)
                    refLine_cords = observation_dict[obj_refLine_key]
                    x_peak, y_peak = int(refLine_cords[0]), int(refLine_cords[1])
                    GridAxis_list[i].scatter(x_peak, y_peak, s=30, facecolor='green')

                    # Loading the cropping area
                    obj_crop_key = '{Color}_cropping'.format(Color=arm_color)
                    crop_cords = map(int, observation_dict[obj_crop_key])
                    GridAxis_list[i].add_patch(
                        patches.Rectangle((crop_cords[0], crop_cords[2]), crop_cords[1] - crop_cords[0],
                                          crop_cords[3] - crop_cords[2], linewidth=2, color='red',
                                          fill=False))  # remove background

                    # Loading the scaling area area
                    obj_scaling_key = '{Color}_scale_region'.format(Color=arm_color)
                    scale_cords = map(int, observation_dict[obj_scaling_key])
                    GridAxis_list[i].add_patch(
                        patches.Rectangle((scale_cords[0], scale_cords[2]), scale_cords[1] - scale_cords[0],
                                          scale_cords[3] - scale_cords[2], linewidth=2, color='black',
                                          fill=False))  # remove background

            #                     try:
            #                     #Print the maxima
            #                     if frames_color[index_i] == 'Blue arm':
            #                         region_zone =   self.default_OIII5007_region
            #                         scale_zone  =   self.default_blue_Scaleregion
            #                     elif frames_color[index_i] == 'Red arm':
            #                         region_zone =   self.default_SIII9531_region
            #                         scale_zone  =   self.default_red_Scaleregion
            #
            #                         section         = image_data[region_zone[0]:region_zone[1],region_zone[2]:region_zone[3]]
            #                         max_value_sec   = npmax(section)
            #                         max_indeces_sec = where(image_data == max_value_sec)
            #
            #                         new_limit           = max_indeces_sec[0][0] - 50
            #                         section_2           = image_data[region_zone[0]:new_limit, region_zone[2]:region_zone[3]]
            #                         max_value_sec2      = npmax(section_2)
            #                         max_indeces_sec2    = where(image_data == max_value_sec2)
            #
            #                         #plotting the points
            #                         GridAxis_list[i].clear()
            #
            #                         #Get zscale limits for plotting the image
            #                         IntensityLimits     = ZScaleInterval()
            #                         int_min, int_max    = IntensityLimits.get_limits(image_data)[0], IntensityLimits.get_limits(image_data)[1]
            #                         cmap_frame          = self.frames_colors[frames_color[index_i]]
            #                         GridAxis_list[i].imshow(image_data, cmap=cmap_frame, origin='lower', vmin = int_min, vmax = int_max)
            #
            #                         GridAxis_list[i].scatter(max_indeces_sec[1], max_indeces_sec[0], s=40, edgecolor='black', facecolor='none')
            #                         GridAxis_list[i].scatter(max_indeces_sec2[1], max_indeces_sec2[0], s=40, edgecolor='yellow', facecolor='none')
            #                         GridAxis_list[i].text(max_indeces_sec[1] + 10, max_indeces_sec[0], '{x} {y}'.format(x =max_indeces_sec[1], y=max_indeces_sec[0]), fontsize=16)
            #                         GridAxis_list[i].text(max_indeces_sec2[1] + 10, max_indeces_sec2[0], '{x} {y}'.format(x =max_indeces_sec2[1], y=max_indeces_sec2[0]), fontsize=16)
            #                         GridAxis_list[i].add_patch(patches.Rectangle((scale_zone[2], scale_zone[0]), scale_zone[3] - scale_zone[2], scale_zone[1] - scale_zone[0], fill=False))      # remove background
            #
            #                         GridAxis_list[i].set_title(files_name[index_i], fontsize = 20)
            #                         GridAxis_list[i].axes.get_yaxis().set_visible(False)
            #                         GridAxis_list[i].axes.get_xaxis().set_visible(False)
            #
            #                     except:
            #                         print 'Could not print'

            # Add the plot
            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(height=NoEscape(r'1\textheight'))
                doc.append(NoEscape(r'\newpage'))
                plt.cla()

        doc.generate_pdf(clean_tex=True)

        return

    def spectra(self, pdf_address, indeces_frame, ext=0):

        files_name = self.reducDf[indeces_frame].sort_values(['frame_tag']).file_name.values
        files_address = self.reducDf[indeces_frame].sort_values(['frame_tag']).file_location.values
        frames_color = self.reducDf[indeces_frame].sort_values(['frame_tag']).ISIARM.values
        Number_frames = len(files_name)

        # Create the doc
        doc = Document(pdf_address)
        doc.packages.append(
            Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=1cm', 'landscape']))
        doc.append(NoEscape(r'\extrafloats{100}'))
        doc.append(NoEscape(r'\pagenumbering{gobble}'))

        # Loop through the fits data:
        for i in range(Number_frames):

            wavelength, Flux_array, Header_0 = self.get_spectra_data(files_address[i] + files_name[i])

            # Define image
            Fig = plt.figure(figsize=(16, 10))
            Axis = Fig.add_subplot(111)

            Axis.set_xlabel(r'Wavelength $(\AA)$', fontsize=20)
            Axis.set_ylabel('Flux' + r'$(erg\,cm^{-2} s^{-1} \AA^{-1})$', fontsize=20)
            Axis.set_title(files_name[i] + ' extracted spectrum', fontsize=25)

            if ('NAXIS2' in Header_0) and ('NAXIS3' in Header_0):
                appertures_number = Header_0['NAXIS2']
                spectra_number = 1
                for apper_n in range(appertures_number):
                    Axis.plot(wavelength, Flux_array[0][apper_n],
                              label='extracted spectrum apperture: {apper_n}'.format(apper_n=apper_n + 1))
            elif ('NAXIS2' in Header_0) and ('NAXIS1' in Header_0):
                appertures_number = Header_0['NAXIS2']
                spectra_number = 1
                for apper_n in range(appertures_number):
                    Axis.plot(wavelength, Flux_array[apper_n],
                              label='extracted spectrum apperture: {apper_n}'.format(apper_n=apper_n + 1))
            else:
                spectra_number = 1
                appertures_number = 1
                Axis.plot(wavelength, Flux_array, label='extracted spectrum')

            Axis.legend()

            with doc.create(Figure(position='htbp')) as plot:

                #                     plot.add_plot(height=NoEscape(r'1\textheight'))
                plot.add_plot(width=NoEscape(r'1\textwidth'))

                # Reset the image
                doc.append(NoEscape(r'\newpage'))
                plt.clf()

        doc.generate_pdf(clean_tex=True)

        return
