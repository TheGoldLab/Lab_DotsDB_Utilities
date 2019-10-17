"""
The goal of this module is to provide a standard API to manage databases of random dots stimuli.
The format for the databases is HDF5. The main functionalities are:

1. create databases of random dots stimuli
    either by generating the stimuli
    or by providing existing stimuli
2. append data to existing db
3. inspect db (including parameters and other metadata about the stimuli)
4. extract data from db

Logic
-----

The idea is to generate the stimulus with a :class:`DotsStimulus` instance and then write it to file with the
:func:`write_stimulus_to_file` function.

A word on pixel frames and their HDF5 storage
+++++++++++++++++++++++++++++++++++++++++++++

Within this module, every pixel frame is handled as a 2D numpy array of booleans. The (i,j) entry in the array
corresponds to the pixel at position (i,j) in space.

.. note::
    For pixel with coordinates (i,j), small i corresponds to top, large i to bottom. Small y corresponds to left, \
    large y to right.

Details on how such pixel frame is written to HDF5 file may be found in the documentation for the \
:func:`write_stimulus_to_file` function

"""
import numpy as np
import pandas as pd
import h5py
import time
import sys


"""--------------------- DISPLAY FUNCTIONS ------------------------"""


def wrap_norm_coord(to_wrap):
    """
    if to_wrap is not in [0,1), wraps it around
    :param to_wrap: a number
    :return: decimal part of the number
    """
    return np.mod(to_wrap, 1)


def point_to_pixel(coord, grid_dims):
    """
    converts the coordinates of a dot from normalized space to pixel space

    .. note::
        Recall, whether normalized or in pixel space, with coordinates of the form (x,y), x represents the vertical \
        position and y the horizontal position. Small x corresponds to top, large x corresponds to bottom. \
        Small y corresponds to left, large y corresponds to right.

    :param coord: ndarray of length 2 with double entries representing normalized coordinates of a dot
    :param grid_dims: dimensions of grid in pixels, e.g. np.array((3,2)) for a 3 x 2 grid
    :type grid_dims: ndarray of length 2 with integer entries
    :return: ndarray of length 2 and entries of
    :rtype: np.intp
    """
    px_indices = np.floor(grid_dims * coord)
    return px_indices.astype(np.intp)


def pixel_to_patch(px_idxs, patch_dims, grid_dims):
    """
    Converts the coords of the center of a dot in pixel space into a patch of pixels.
    A patch is completely determined by the position of its top left corner and its dimensions.
    Even vs. odd values of the entries of patch_dims are treated differently.
    Parts of patch falling outside the grid are truncated.

    .. note::
        Right now, only square grids and square patches are handled.

    :param px_idxs: pixel coordinates of a dot
    :param patch_dims: 2-tuple containing height and width of patch in pixels
    :param grid_dims: height and width of pixel grid
    :return: top left corner of patch and its dims.
    """
    h = patch_dims[0]  # patch height in px
    w = patch_dims[1]
    if h == w and grid_dims[0] == grid_dims[1]:  # if patch and grid have square shape
        max_px_idx = grid_dims[1] - 1
        if np.mod(h, 2):  # if patch height is an odd number of pixels
            pxs_to_padd = (h - 1) / 2

            top_left_corner = px_idxs - pxs_to_padd
            top_left_corner[top_left_corner < 0] = 0

            bottom_right_corner = px_idxs + pxs_to_padd
            bottom_right_corner[bottom_right_corner > max_px_idx] = max_px_idx

            new_patch_dims = bottom_right_corner - top_left_corner + 1
        else:
            pxs_to_padd_top = (h / 2) - 1
            pxs_to_padd_bottom = pxs_to_padd_top + 1
            pxs_to_padd_left = pxs_to_padd_top
            pxs_to_padd_right = pxs_to_padd_bottom

            top_left_corner = px_idxs - np.array([pxs_to_padd_top, pxs_to_padd_left])
            top_left_corner[top_left_corner < 0] = 0

            bottom_right_corner = px_idxs + np.array([pxs_to_padd_bottom, pxs_to_padd_right])
            bottom_right_corner[bottom_right_corner > max_px_idx] = max_px_idx

            new_patch_dims = bottom_right_corner - top_left_corner + 1

    return top_left_corner.astype(np.intp), new_patch_dims.astype(np.intp)


def set_patch(patch_top_left_corner, patch_dims, grid, value=np.True_):
    """
    Sets the values in grid that correspond to the patch location to np.True_

    :param patch_top_left_corner: pixel coordinates of top corner of patch
    :param patch_dims: 2-ndarray in pixels
    :type patch_dims: np.intp
    :param grid: 2D ndarray
    :param value: value to what pixels should be set (default=np.True_)
    :return: grid is changed in-place
    :rtype: None
    """
    x = patch_top_left_corner[0]
    y = patch_top_left_corner[1]
    x_end = x + patch_dims[0]
    y_end = y + patch_dims[1]

    grid[x:x_end, y:y_end] = value

    return None


def flatten_pixel_frame(f):
    """
    a 2D pixel frame is flattened, whereby **rows are concatenated**

    :param f: 2D array representing a pixel frame
    :type f: ndarray of boolean values
    :rtype: 1D ndarray of boolean values
    """
    return f.reshape(f.size)


"""----------------------- DB FUNCTIONS ------------------------"""


def write_stimulus_to_file(stim, num_of_trials, filename, create_file=True, append_to_group=False,
                           pre_generated_stimulus=None, group_name=None):
    """
    Write the stimulus generated by (or attached to) a :class:`DotsStimulus` instance to HDF5 file.
    A specific group in the file is created, with group name generated by :func:`build_group_name`.
    Then, the *px* dataset is created within that group. Such dataset has a 2D shape.
    Each row is a flattened frame (see :func:`flatten_pixel_frame`)

    :param stim: an instance of DotsStimulus
    :param num_of_trials: number of trials to generate
    :param filename: path including file name with extension
    :type filename: str
    :param create_file: whether to create a new file or not (default=True)
    :type create_file: bool
    :param append_to_group: if true (this is not implemented yet), will append the data to an existing dataset \
    (default=False)
    :type append_to_group: bool
    :param pre_generated_stimulus: each entry in this list will be used as DotsStimulus.attached_data.
    :type pre_generated_stimulus: list of lists of numpy arrays.
    :rtype: None
    """
    stim_params = stim.export_params()

    if create_file:
        f = h5py.File(filename, 'x')  # create file, fails if exists
    else:
        f = h5py.File(filename, 'r+')  # open in read/write mode, fails if file doesn't exist

    # create group corresponding to parameters, if not provided
    if group_name is None:
        group_name = build_group_name(stim)

    if not append_to_group:
        if group_name in f:
            raise ValueError(f"group {group_name} already exists in file")
        group = f.create_group(group_name)
    else:
        raise NotImplementedError('this feature is not implemented yet')

    # and add all parameters as attributes
    for k, v in stim_params.items():
        group.attrs.__setitem__(k, v)

    # generate stimulus upfront
    use_pre_generated_stimulus = (pre_generated_stimulus is not None)
    if use_pre_generated_stimulus:
        assert len(pre_generated_stimulus) == num_of_trials

    # create dataset with variable length (because each trial may have a different number of frames)
    vlen_data_type = h5py.special_dtype(vlen=np.bool_)
    dset = group.create_dataset("px",
                                (num_of_trials,),
                                compression="gzip",
                                compression_opts=9,
                                fletcher32=True,
                                dtype=vlen_data_type)

    for t in range(num_of_trials):
        if use_pre_generated_stimulus:
            stim.attached_data = pre_generated_stimulus[t]

        frames_seq = [flatten_pixel_frame(
            stim.norm_to_pixel_frame(fr)
            ) for fr in list(stim.normalized_dots_frame_generator())]
        dset[t] = np.concatenate(frames_seq, axis=None)

    f.flush()
    f.close()
    return None


def dump_trial_as_jpg(trial_array, file_path):
    """
    Writes each frame from a trial as a separate .jpg file

    To lump all files into a single trial.mp4 file from Linux Terminal, type:
    $ ffmpeg -framerate 60 -i frame-%0d.jpg -c:v libx264 -profile:v high -crf 20 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p trial.mp4

    :param trial_array: numpy.ndarray as returned by :func:`extract_trial_as_3d_array`
    :param file_path: (str) path up to folder included where images should be created
    :return:
    """
    import imageio
    num_frames = trial_array.shape[2]
    for i in range(num_frames):
        imageio.imwrite(file_path + 'frame-' + str(i) + '.jpg',
                        trial_array[:, :, i].transpose().astype(int))


def inspect_db(filename):
    """
    Loops recursively through all groups and datasets of an hdf5 database, and
    displays their attributes (and shape for datasets)
    file is opened in read-only mode with context manager

    :param filename: full path to hdf5 file
    :return: dict with, as top-level keys, the names of the h5py objects contained in the file
    """
    def _inspect_element(name):
        """
        Important, for this function to work, an h5py.File instance must be in the
        scope of the function with name ff, and a dict with name dd as well

        :param name: name of hdf5 object
        :return:
        """
        obj = ff[name]  # current object being visited
        attributes = [(k, v) for (k, v) in obj.attrs.items()]
        if isinstance(obj, h5py.Dataset):
            dd[name] = {'type': 'dataset', 'attrs': attributes, 'shape': obj.shape}
        elif isinstance(obj, h5py.Group):
            dd[name] = {'type': 'group', 'attrs': attributes}
        else:
            print(f"visited object {name} is neither group nor dataset, type is {type(obj)}")

        return None

    dd = {}  # initialize dict to return

    with h5py.File(filename, 'r') as ff:
        ff.visit(_inspect_element)

    return dd


def extract_trial_as_3d_array(path_to_file, dset_name, group_name, trial_number):
    """
    extracts a specific trial from the database and returns it as a 3D matrix of pixel frames
    Uses context manager to open the file in read-only mode
    TODO: better explain the orientation of the dots (what is left-right, what is bottom-up)

    :param path_to_file: string with absolute path to hdf5 file
    :param dset_name: string for name of dataset in hdf5 file (full path within the file)
    :param group_name: string for name of group that contains the dataset in hdf5 file (full path within the file)
    :param trial_number: number for the trial to extract, should be an int >= 1
    :return: 3D numpy array, first two dims for pixels, last dim for frames
    """

    with h5py.File(path_to_file, 'r') as f:
        s = f[dset_name]
        g = f[group_name]
        assert (isinstance(s, h5py.Dataset) and isinstance(g, h5py.Group))
        assert 1 <= trial_number <= s.shape[0]
        trial = s[trial_number - 1]

        npx = g.attrs['frame_width_in_pxs']
#        nf = g.attrs['num_frames']
        nf = trial.size / (npx**2)
#        print(npx, nf, (npx**2)*nf, s.shape, trial.shape, trial.size)
        assert nf.is_integer()
        nf = int(nf)

    return trial.reshape((npx, npx, nf), order="F")


# def create_hdf5_data_structure(hdf5file, groupname, num_trials, num_samples, max_trials=1000000):
#     """
#     :param hdf5file: h5py.File
#     :param groupname:
#     :param num_trials: nb of trials
#     :param max_trials: for db decision datasets max nb of rows
#     :param num_samples: for db decision datasets; nb of cols
#     :return: created group
#     """
#     group = hdf5file.create_group(groupname)
#     dt = h5py.special_dtype(vlen=np.dtype('f'))
#     group.create_dataset('trials', (num_trials, 3), maxshape=(max_trials, 10), dtype=dt)
#     group.create_dataset('trial_info', (num_trials, 3), maxshape=(max_trials, 10), dtype='f')
#     group.create_dataset('decision_lin', (num_trials, num_samples), dtype='i', maxshape=(max_trials, num_samples))
#     group.create_dataset('decision_nonlin', (num_trials, num_samples),
#                          dtype='i', maxshape=(max_trials, num_samples))
#     return group
#
#
# def populate_hdf5_db(fname, four_par, num_of_trials, number_of_samples=1):
#     """
#     Generate stimulus data and store as hdf5 file.
#     This is the main function called by this script.
#     """
#     # open/create file
#     f = h5py.File(fname, 'a')
#     ll, lh, h, t = four_par
#
#     # create group corresponding to parameters
#     group_name = build_group_name(four_par)
#
#     if group_name in f:  # if dataset already exists, exit without doing anything
#         print('data already present, file left untouched')
#     else:  # if dataset doesn't exist, create it
#         print('creating dataset with group name {}'.format(group_name))
#         grp = create_hdf5_data_structure(f, group_name, num_of_trials, num_samples=number_of_samples)
#
#         # create trials dataset
#         trials_data = grp['trials']
#         # get row indices of new data to insert
#         row_indices = np.r_[:num_of_trials]
#
#         # create info on data
#         info_data = grp['trial_info']  # info dataset
#         info_data.attrs['h'] = h
#         info_data.attrs['T'] = t
#         info_data.attrs['low_click_rate'] = ll
#         info_data.attrs['high_click_rate'] = lh
#         info_data.attrs['S'] = (lh - ll) / np.sqrt(ll + lh)
#         data_version = 1  # version number of new data to insert
#
#     # populate database
#     for row_idx in row_indices:
#         # vector of CP times
#         cptimes = gen_cp(t, h)
#         trials_data[row_idx, 2] = cptimes
#
#         # stimulus (left clicks, right clicks)
#         (left_clicks, right_clicks), init_state, end_state = gen_stim(cptimes, ll, lh, t)
#         trials_data[row_idx, :2] = left_clicks, right_clicks
#
#         # populate info dataset
#         info_data[row_idx, :] = init_state, end_state, data_version
#
#     info_data.attrs['last_version'] = data_version
#     f.flush()
#     f.close()
#
#
# def dump_info(four_parameters, s, nt, nruns):
#     print('S value: {}'.format(s))
#     print('low click rate: {}'.format(four_parameters[0]))
#     print('high click rate: {}'.format(four_parameters[1]))
#     print('hazard rate: {}'.format(four_parameters[2]))
#     print('interr. time: {}'.format(four_parameters[3]))
#     print('nb of trials / hist: {}'.format(nruns))
#     print('nb of trials in sequence: {}'.format(nt))


def build_group_name(stimulus):
    """
    Grabs the parameters from the given :class:`DotsStimulus` instance and builds a string out of them

    :param stimulus: a :class:`DotsStimulus` object
    :rtype: str
    """
    params = stimulus.export_params()
    abbrev = {
        'interleaves': 'intlv',
        'limit_life_time': 'lft',
        'frame_rate': 'fr',
        'field_scale': 'fs',
        'speed': 'sp',
        'density': 'ds',
        'coh_mean': 'c',
        'coh_stdev': 'cs',
        'direction': 'd',
        'num_frames': 'nf',
        'diameter': 'dm',
        'stencil_radius_in_vis_angle': 'sc',
        'pixels_per_degree': 'ppd',
        'dot_size_in_pxs': 'dts',
        'frame_width_in_pxs': 'fw'
    }
    return '_'.join(['%s%s' % (abbrev[key], value) for (key, value) in params.items()])

#
# def update_linear_decision_data(file_name, group_name, num_samples, sample_range, create_nonlin_db=False):
#     """
#     :param file_name: file name (string)
#     :param group_name: group object from h5py module
#     :param num_samples:
#     :param sample_range: (starting value, ending value)
#     :param create_nonlin_db:
#     :return:
#     """
#     f = h5py.File(file_name, 'r+')
#     group = f[group_name]
#     info_dset = group['trial_info']
#     trials_dset = group['trials']
#     num_trials = trials_dset.shape[0]
#     row_indices = range(num_trials)
#     dset_name = 'decision_lin'
#     if create_nonlin_db:
#         # create dataset for nonlinear decisions
#         group.create_dataset('decision_nonlin', (num_trials, num_samples),
#                              dtype='i', maxshape=(100000, 10001))
#     dset = group[dset_name]
#
#     # store best gamma as attribute for future reference if doesn't exist
#     skellam = info_dset.attrs['S']
#     h = info_dset.attrs['h']
#     if 'best_gamma' in dset.attrs.keys():
#         best_gamma = dset.attrs['best_gamma']
#     else:
#         if skellam in np.arange(0.5, 10.1, 0.5) and h == 1:
#             best_gamma = get_best_gamma(skellam, h, polyfit=False)
#         else:
#             best_gamma = get_best_gamma(skellam, h)
#         dset.attrs['best_gamma'] = best_gamma
#     gamma_samples, gamma_step = np.linspace(sample_range[0], sample_range[1], num_samples, retstep=True)
#     attrslist = ['init_sample', 'end_sample', 'sample_step']
#     values_dict = {'init_sample': sample_range[0],
#                    'end_sample': sample_range[1],
#                    'sample_step': gamma_step}
#     for attrname in attrslist:
#         if attrname not in dset.attrs.keys():
#             dset.attrs[attrname] = values_dict[attrname]
#
#     # populate dataset
#     for row_idx in row_indices:
#         stim = tuple(trials_dset[row_idx, :2])
#         gamma_array = np.reshape(np.r_[best_gamma, gamma_samples], (-1, 1))
#         dset[row_idx, :] = decide_linear(gamma_array, stim)
#     f.flush()
#     f.close()


"""--------------------- STIMULUS CLASS ------------------------"""


class DotsStimulus:
    """
    Object used to handle a single trial of the random dots stimulus.
    A trial consists in a sequence of frames.

    .. note::
        Right now, the main way to generate a stimulus is to use the generator \
        :func:`DotsStimulus.normalized_dots_frame_generator`
    """
    interleaves = 3
    """size of the cycle of interleaved frames"""
    limit_life_time = True
    """detail"""
    frame_rate = 60  # should be a multiple of 10
    """rate in Hz at which the stimulus should be interpreted"""
    field_scale = 1.1
    """scalar by which the drawing aperture is multiplied to define actual pixel space"""

    def __init__(self,
                 speed,
                 density,
                 coh_mean,
                 coh_stdev,
                 direction,
                 num_frames,
                 diameter,
                 stencil_radius_in_vis_angle=None,
                 pixels_per_degree=55.4612,
                 dot_size_in_pxs=6,
                 attached_data=None):
        """
        :param speed: speed of dots, in deg vis. angle per sec
        :param density: nb of dots per unit area, where distance unit is deg vis. angles
        :param coh_mean: between 0 and 100, mean percentage of coherently moving dots
        :param coh_stdev: stdev of percentage of coherently moving dots
        :param direction: 'right' or 'left' or 'last_right' or 'last_left'
        :param num_frames: number of frames for stimulus
        :param diameter: in deg vis angle
        :param stencil_radius_in_vis_angle: radius of stencil that encloses visible dots (defaults to diameter / 2)
        :param attached_data: a list of normalized dots frames.
            this data will be used by the normalized_dots_frame_generator. If None, data will be generated)
        :type attached_data: None or list of numpy arrays, each of shape (n, 2); where n may vary, it is the number of
            dots in the frame. Note, at this stage, dots falling outside the stencil disk need not be trimmed. The
            trimming is handled by the norm_to_pixel_frame method.
        """
        self.speed = speed

        # controls total number of dots across self.interleaves frames
        self.density = density

        # to further randomize coherence level on each frame
        self.coh_mean = coh_mean  # should be between 0 and 100
        self.coh_stdev = coh_stdev

        # total number of frames in stimulus
        self.num_frames = num_frames

        # diameter of region on screen in degrees of visual angle, in which dots appear
        self.diameter = diameter

        # actual width of field in screen in which dots are drawn.
        # in theory, a dot outside of the allowed stimulus region is invisible,
        # but this is not yet implemented...
        self.field_width = self.diameter * self.field_scale

        if stencil_radius_in_vis_angle is None:
            self.stencil_radius_in_vis_angle = self.diameter / 2
        else:
            self.stencil_radius_in_vis_angle = stencil_radius_in_vis_angle
        self.stencil_radius_in_norm_units = self.stencil_radius_in_vis_angle / self.field_width

        # some useful pixel dimensions
        self.pixels_per_degree = pixels_per_degree
        self.dot_size_in_pxs = dot_size_in_pxs
        self.frame_width_in_pxs = np.floor(self.pixels_per_degree * self.field_width).astype(int)

        # step size for coherent dots displacement (displacement is applied every self.interleaves frame)
        self.coh_step = self.speed / self.field_width * (self.interleaves / self.frame_rate)
        # set negative step if direction of motion is left
        self.direction = direction
        if self.direction == 'left':
            self.coh_step = - self.coh_step

        # total number of dots across self.interleaves frames
        self.num_dots_in_chunk = np.ceil(self.density * self.field_width**2 / self.frame_rate)

        # list of self.interleaves indexing arrays (mainly used for lifetimes)
        # each entry in the list below acts as an indexing array, and it is meant to only
        # selects the dots from a particular frame, according to its interleaved position
        self.dots_idxs = \
            [np.arange(n, self.num_dots_in_chunk, self.interleaves).astype(np.intp)
             for n in range(self.interleaves)]

        # list of self.interleaves dot counts
        self.num_dots_in_frames = [x.size for x in self.dots_idxs]
        # alternative way of computing the oneliner above:
        # self.num_dots_in_frames = [self.num_dots_in_chunk // self.interleaves] * self.interleaves
        # for idx in range(self.num_dots_in_chunk % self.interleaves):
        #     self.num_dots_in_frames[idx] += 1

        self._attached_data = attached_data

    @property
    def attached_data(self):
        return self._attached_data

    @attached_data.setter
    def attached_data(self, value):
        """just to make sure that self.num_frames is updated when new data is attached"""
        self.num_frames = len(value)
        self._attached_data = value

    def export_params(self):
        """export parameters of the stimulus"""
        return {
            'interleaves': self.interleaves,
            'limit_life_time': self.limit_life_time,
            'frame_rate': self.frame_rate,
            'field_scale': self.field_scale,
            'speed': self.speed,
            'density': self.density,
            'coh_mean': self.coh_mean,
            'coh_stdev': self.coh_stdev,
            'direction': self.direction,
            'num_frames': self.num_frames,
            'diameter': self.diameter,
            'stencil_radius_in_vis_angle': self.stencil_radius_in_vis_angle,
            'pixels_per_degree': self.pixels_per_degree,
            'dot_size_in_pxs': self.dot_size_in_pxs,
            'frame_width_in_pxs': self.frame_width_in_pxs
        }

    def normalized_dots_frame_generator(self, max_frames=None):
        """
        Create a *generator* to generate frames of a random dots stimulus.

        Each frame is a numpy array with N rows and 2 columns.
        Each row corresponds to the *normalized* coordinates of a dot: col 1 corresponds to the vertical
        position of the dot, col 2 corresponds to the horizontal position of the dot. For example,
        (0,0) is the top left corner; (.99, .99) is the bottom right corner;
        (0,.5) is a dot positioned at the top row and midway between left and right.
        The first frame of each interleaved sequence is randomly generated
        
        .. note::
            1 is not an allowed coordinate, only values in [0,1) are.

        :param max_frames: max number of frames to generate (defaults to self.num_frames)
        :return: Yields a 'successor' frame on each iteration.
        :rtype: generator
        """
        if max_frames is None:
            max_frames = self.num_frames

        if self.attached_data is None:
            frame_count = 0
            lifetimes = np.zeros(self.num_dots_in_chunk.astype(int))
            frame_chunk = [np.random.rand(self.num_dots_in_frames[n], 2) for n in range(self.interleaves)]
            while frame_count < max_frames:
                mod_idx = np.mod(frame_count, self.interleaves)
                ancestor = frame_chunk[mod_idx]
                new_frame, updated_lifetimes = self.next_frame(ancestor, lifetimes[self.dots_idxs[mod_idx]])
                lifetimes[self.dots_idxs[mod_idx]] = updated_lifetimes
                frame_chunk[mod_idx] = new_frame
                yield new_frame
                frame_count += 1
        else:
            counter = 1
            for fr in self.attached_data:
                if counter > max_frames:
                    break
                yield fr
                counter += 1

    def next_frame(self, present_frame, present_lifetimes):
        """
        Computes the next frame. This function mimicks dotsDrawableDotKinetogram.computeNextFrame()
        from the `MATLAB code <https://github.com/TheGoldLab/Lab-Matlab-Control/blob/eyeDev/snow-dots/classes/drawable/dotsDrawableDotKinetogram.m>`_

        :param present_frame: normalized frame as handled by self.normalized_dots_frame_generator()
        :param present_lifetimes: numpy array
        :return: tuple (successor frame, new lifetimes)
        """
        # number of dots in current frame
        num_dots = present_frame.shape[0]

        # select coherence stochastically, and cap at 100% max
        coherence = np.abs(np.random.normal(self.coh_mean, self.coh_stdev))
        if coherence > 100:
            coherence = 100

        # select number of coherent dots in this frame stochastically
        num_coh_dots = np.random.binomial(num_dots, coherence / 100)

        if self.limit_life_time:
            # easier to use pandas for manipulations to come
            compound_data = {
                'x': present_frame[:, 0],
                'y': present_frame[:, 1],
                'lifetime': present_lifetimes,
                'is_coherent': np.full_like(present_lifetimes, np.False_, dtype=np.bool_)
            }

            dots_dataframe = pd.DataFrame(compound_data)
            lifetime_ordered_dots = dots_dataframe.sort_values(by=['lifetime'])

            # set dots with smallest life times as coh dots, remaining ones as non-coh
            # following syntax was hard to find. I found it here:
            # https://stackoverflow.com/a/44792367
            lifetime_ordered_dots.iloc[:num_coh_dots,
                                       lifetime_ordered_dots.columns.get_loc('is_coherent')] = np.True_

            # increment lifetime by 1 for coherent dots
            lifetime_ordered_dots.loc[lifetime_ordered_dots['is_coherent'], 'lifetime'] += 1

            # set lifetime of non-coherent dots to 0
            lifetime_ordered_dots.loc[np.logical_not(lifetime_ordered_dots['is_coherent']),
                                      'lifetime'] = 0

            # extract new life times in initial dots order to return
            new_life_times = np.array(lifetime_ordered_dots.sort_index()['lifetime'])

            # update dots positions
            # draw random positions for non-coherent dots
            lifetime_ordered_dots.loc[np.logical_not(lifetime_ordered_dots['is_coherent']),
                                      ['x', 'y']] = np.random.rand(num_dots - num_coh_dots, 2)

            # update horizontal position of coherent dots with appropriate coherent motion step
            lifetime_ordered_dots.loc[lifetime_ordered_dots['is_coherent'],
                                      'y'] += self.coh_step

            # draw random positions for vertical position of dots which will be wrapped
            num_wrap = lifetime_ordered_dots.loc[
                (lifetime_ordered_dots['y'] < 0) | (lifetime_ordered_dots['y'] >= 1),
                'x'].count()
            lifetime_ordered_dots.loc[
                (lifetime_ordered_dots['y'] < 0) | (lifetime_ordered_dots['y'] >= 1),
                'x'] = np.random.rand(num_wrap)

            # wrap around horizontally dots who fell outside
            lifetime_ordered_dots['y'] = lifetime_ordered_dots['y'].apply(wrap_norm_coord)

            successor_frame = np.array(lifetime_ordered_dots.sort_index()[['x', 'y']])

        return successor_frame, new_life_times

    def norm_to_pixel_frame(self, normalized_frame):
        """
        takes a normalized frame of dots as input and returns a pixel frame
        A dot in normalized coordinates is characterized by a single pair of doubles.
        However, in pixel space, a dot is a square patch of pixels.
        Dots falling at the edge of the pixel space are truncated, so not all dots have necessarily the same number of
        active pixels.

        :param normalized_frame: num_dots x 2 array
        :type normalized_frame: ndarray of doubles
        :return: pixel frame represented as an num_pixels x num_pixels ndarray with np.bool_ entries
        """
        # create square grid of pixels and define dimensions of dot in pixel space
        grid_size = self.frame_width_in_pxs
        grid = np.full((grid_size, grid_size), np.False_)  # array of boolean values representing pixels
        patch_shape = (self.dot_size_in_pxs, self.dot_size_in_pxs)

        # loop over dots and set the corresponding pixels to np.True_
        for point in normalized_frame:
            # if dot falls outside visible region (stencil), do not draw it
            if np.sum((point-0.5)**2) <= self.stencil_radius_in_norm_units**2:
                pt_coord_in_pxs = point_to_pixel(point, grid.shape)
                corner, dims = pixel_to_patch(pt_coord_in_pxs,  patch_shape, grid.shape)
                set_patch(corner, dims, grid)
        return grid


if __name__ == '__main__':
    """
    Script may be called with 0 or 6 arguments from command line. With 6 arguments, these are:
    1. speed
    2. direction
    3. coherence
    4. num_trials
    5. num_frames
    6. db filename

    With 0 arguments, the script appends requested datasets to given file 
    """
    if len(sys.argv) == 7:
        # speed
        try:
            sp = float(sys.argv[1])
            assert(sp > 0)
        except ValueError:
            print('\nError msg: first command line arg corresponding to speed should be a positive scalar\n')
            exit(1)

        # direction
        try:
            dir = sys.argv[2]
            assert(dir == "left" or dir == "right" or dir == "last_left" or dir == "last_right")
        except ValueError:
            print('\nError msg: second command line arg corresponding to direction should be either "left" or "right"'
                  'or "last_left" or "last_right"\n')
            exit(1)

        # coherence
        try:
            coh = float(sys.argv[3])
            assert(coh >= 0 and coh <= 100)
        except ValueError:
            print('\nError msg: third command line arg corresponding to coherence should be a non-negative scalar'
                  'between 0 and 100\n')
            exit(1)

        # Number of trials
        try:
            num_trials = int(sys.argv[4])
            assert(num_trials > 0)
        except ValueError:
            print('\nError msg: fourth command line arg corresponding to number of trials should be a positive integer\n')
            exit(1)

        # Number of frames
        try:
            num_of_frames = int(sys.argv[5])
            assert(num_of_frames > 0)
        except ValueError:
            print('\nError msg: fifth command line arg corresponding to number of frames should be a positive integer\n')
            exit(1)

        # hdf5 db filename
        try:
            db_filename = sys.argv[6]
            if db_filename[-3:] != '.h5':
                raise ValueError("By convention, db filename should end with '.h5'")
        except ValueError as err:
            print('\nError msg: sixth command line arg corresponding to filename has a pb')
            print(err.args)
            exit(1)

        start_time = time.time()

        parameters = dict(
            speed=sp,
            density=90,
            coh_mean=coh,
            coh_stdev=10,
            direction=dir,
            num_frames=num_of_frames,
            diameter=5
        )

        stimulus = DotsStimulus(**parameters)
        write_stimulus_to_file(stimulus, num_trials, db_filename)

        print("--- {} seconds ---".format(time.time() - start_time))

    elif len(sys.argv) == 1:
        start_time = time.time()
        # this script adds new groups (with newly generated datasets) to an existing dotsDB HDF5 file

        # file name
        file_name = '../data/test2.h5'

        # parameters of new datasets to create:
        params = {
            'speed': [5],
            'density': [90],
            'coh_mean': [0, 30, 80],
            'coh_stdev': [10],
            'direction': ['left', 'right'],
            'num_frames': [15],
            'diameter': [5]
        }

        num_trials = 50

        # edit params so that shorter entries are recycled

        # get total number of combinations
        num_comb = 1
        for v in params.values():
            num_comb *= len(v)

        # recycle values 
        for k, v in params.items():
            recycle_factor = num_comb // len(v)
            params[k] *= recycle_factor

        import pprint
        #pprint.pprint(params)
        #print(num_trials)

        for dset_idx in range(num_comb):
            curr_dict = {k: v[dset_idx] for k, v in params.items()}
            S = DotsStimulus(**curr_dict)
            write_stimulus_to_file(S, num_trials, file_name, create_file=(dset_idx == 0)) # only create the file at first iteration

        print("--- {} seconds ---".format(time.time() - start_time))
        
        pprint.pprint(inspect_db(file_name), width=120)

    else:
        raise OSError('Script called with wrong number of command line args')
