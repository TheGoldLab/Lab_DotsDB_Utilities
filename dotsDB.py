import numpy as np
import pandas as pd


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
    :param coord: ndarray of length 2 with double entries representing normalized coordinates of a dot
    :param grid_dims: ndarray of length 2 with integer entries, e.g. np.array((3,2)) for a 3 x 2 grid
    :return: ndarray of length 2 and entries of type np.intp
    """
    px_indices = np.floor(grid_dims * coord)
    return px_indices.astype(np.intp)


def pixel_to_patch(px_idxs, patch_dims, grid_dims):
    """
    Converts the coords of the center of a dot in pixel space into a patch of pixels.
    A patch is completely determined by the position of its top left corner and its dimensions.
    Even vs. odd values of the entries of patch_dims are treated differently.
    Parts of patch falling outside the grid are truncated.
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


def set_patch(patch_top_left_corner, patch_dims, grid, value=True):
    """
    Sets the values in grid that correspond to the patch location to True
    :param patch_top_left_corner: pixel coordinates of top corner of patch
    :param patch_dims: 2-ndarray in pixels (dtype is np.intp)
    :param grid: 2D ndarray
    :param value: value to what pixels should be set
    :return:
    """
    x = patch_top_left_corner[0]
    y = patch_top_left_corner[1]
    x_end = x + patch_dims[0]
    y_end = y + patch_dims[1]

    grid[x:x_end, y:y_end] = value

    return None


def flaten_pixel_frame(f):
    """
    a 2D pixel frame is flattened, whereby rows are concatenated
    :param f: 2D ndarray of boolean values representing a pixel frame
    :return: 1D ndarray of boolean values
    """
    return f.reshape(f.size)


class DotsStimulus:
    interleaves = 3
    limit_life_time = True
    frame_rate = 60  # should be a multiple of 10
    field_scale = 1.1

    def __init__(self, speed, density, coh_mean, coh_stdev, direction, num_frames, diameter,
                 stencil_radius_in_vis_angle=None,
                 pixels_per_degree=55.4612,
                 dot_size_in_pxs=4):
        """
        :param speed:
        :param density:
        :param coh_mean: between 0 and 100, mean percentage of coherently moving dots
        :param coh_stdev: stdev of percentage of coherently moving dots
        :param direction: 'right' or 'left'
        :param num_frames: number of frames for stimulus
        :param diameter: in deg vis angle
        :param stencil_radius_in_vis_angle: defaults to diameter / 2
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

        if stencil_radius_in_vis_angle is None:
            self.stencil_radius_in_vis_angle = self.diameter / 2
        else:
            self.stencil_radius_in_vis_angle = stencil_radius_in_vis_angle
        self.stencil_radius_in_norm_units = self.stencil_radius_in_vis_angle / self.diameter

        # actual width of field in screen in which dots are drawn.
        # in theory, a dot outside of the allowed stimulus region is invisible,
        # but this is not yet implemented...
        self.field_width = self.diameter * self.field_scale

        # step size for coherent dots displacement (displacement is applied every self.interleaves frame)
        self.coh_step = self.speed / self.diameter * (self.interleaves / self.frame_rate)
        # set negative step if direction of motion is left
        if direction == 'left':
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

        self.pixels_per_degree = pixels_per_degree
        self.dot_size_in_pxs = dot_size_in_pxs

    def normalized_dots_frame_generator(self, max_frames=None):
        """
        a frame itself is a numpy array with N rows and 2 columns.
        each row corresponds to a dot, col 1 corresponds to the horizontal
        position of the dot, col 2 corresponds to the vertical position of the dot.
        So, (0,0) is the top left corner; (.99, .99) is the bottom right corner;
        (0,.5) is a dot positioned at the top row and midway between left and right.
        Note that 1 is not an allowed coordinate, only values in [0,1) are.
        :param max_frames: max number of frames to generate (defaults to self.num_frames)
        :return: generator object. Yields a 'successor' frame on each iteration.
        First frame is randomly generated
        """
        if max_frames is None:
            max_frames = self.num_frames
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

    def next_frame(self, present_frame, present_lifetimes):
        """
        Computes the next frame. This function mimicks dotsDrawableDotKinetogram.computeNextFrame()
        from the MATLAB code
        https://github.com/TheGoldLab/Lab-Matlab-Control/blob/eyeDev/snow-dots/classes/drawable/dotsDrawableDotKinetogram.m

        :param present_frame: normalized frame as handled by self.normalized_dots_frame_generator()
        :param present_lifetimes: numpy array
        :return:
        """
        # number of dots in current frame
        num_dots = present_frame.shape[0]

        # select coherence stochastically, and cap at 100% max
        coh = np.abs(np.random.normal(self.coh_mean, self.coh_stdev))
        if coh > 100:
            coh = 100

        # select number of coherent dots in this frame stochastically
        num_coh_dots = np.random.binomial(num_dots, coh / 100)

        if self.limit_life_time:
            # easier to use pandas for manipulations to come
            compound_data = {
                'x': present_frame[:, 0],
                'y': present_frame[:, 1],
                'lifetime': present_lifetimes,
                'is_coherent': np.full_like(present_lifetimes, False, dtype=np.bool)
            }

            dots_dataframe = pd.DataFrame(compound_data)
            lifetime_ordered_dots = dots_dataframe.sort_values(by=['lifetime'])

            # set dots with smallest life times as coh dots, remaining ones as non-coh
            # following syntax was hard to find. I found it here:
            # https://stackoverflow.com/a/44792367
            lifetime_ordered_dots.iloc[:num_coh_dots,
                                       lifetime_ordered_dots.columns.get_loc('is_coherent')] = True

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
        :param normalized_frame: num_dots x 2 ndarray of doubles
        :return: pixel frame represented as an num_pixels x num_pixels ndarray with boolean entries
        """
        # create square grid of pixels and define dimensions of dot in pixel space
        grid_size = np.floor(self.pixels_per_degree * self.field_width).astype(int)
        grid = np.full((grid_size, grid_size), False)  # array of boolean values representing pixels
        patch_shape = (self.dot_size_in_pxs, self.dot_size_in_pxs)

        # loop over dots and set the corresponding pixels to True
        for point in normalized_frame:
            # if dot falls outside visible region (stencil), do not draw it
            if np.sum((point-0.5)**2) <= self.stencil_radius_in_norm_units**2:
                pt_coord_in_pxs = point_to_pixel(point, grid.shape)
                corner, dims = pixel_to_patch(pt_coord_in_pxs,  patch_shape, grid.shape)
                set_patch(corner, dims, grid)
        return grid
