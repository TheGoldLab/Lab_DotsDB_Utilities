import numpy as np
import pandas as pd


def wrap_norm_coord(to_wrap):
    return np.mod(to_wrap, 1)


class DotsStimulus:
    interleaves = 3
    limit_life_time = True
    frame_rate = 60  # should be a multiple of 10
    field_scale = 1.1

    def __init__(self, speed, density, coh_mean, coh_stdev, direction, num_frames, diameter):
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

    def normalized_dots_frame_generator(self):
        """
        a frame itself is a numpy array with N rows and 2 columns.
        each row corresponds to a dot, col 1 corresponds to the horizontal
        position of the dot, col 2 corresponds to the vertical position of the dot.
        So, (0,0) is the top left corner; (.99, .99) is the bottom right corner;
        (0,.5) is a dot positioned at the top row and midway between left and right.
        Note that 1 is not an allowed coordinate, only values in [0,1) are.

        :return: generator object. Yields a 'successor' frame on each iteration.
        First frame is randomly generated
        """
        frame_count = 0
        lifetimes = np.zeros(self.num_dots_in_chunk.astype(int))
        frame_chunk = [np.random.rand(self.num_dots_in_frames[n], 2) for n in range(self.interleaves)]
        while frame_count < self.num_frames:
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


# class DotsFrame:
#     def __init__(self, param_dict):
#         self.num_dots = param_dict['num_dots']
#         self.num_pixels = param_dict['num_pixels']
#         self.pixel_width_of_a_dot = param_dict['pixel_width_of_a_dot']
#         self.pixel_matrix_dims = (round(np.sqrt(self.num_pixels)).astype(int),
#                                   round(np.sqrt(self.num_pixels)).astype(int))
#         self.normalized_representation, self.pixel_representation = self._generate()
#
#     def display(self):
#         print(self.normalized_representation)
#         print(self.pixel_representation)
#
#     def _generate(self, ancestor=None, method='standard'):
#         if ancestor is None:
#             random_coordinates = np.random.rand(self.num_dots, 2)
#             normalized_representation = [NormDot(row[0], row[1]) for row in random_coordinates]
#             print(np.round(normalized_representation, 2))
#         active_pixel_indices = normalized_representation * self.pixel_matrix_dims[0]
#         active_pixel_indices[active_pixel_indices > self.pixel_width_of_a_dot / 2] -= self.pixel_width_of_a_dot / 2
#         active_pixel_indices = np.floor(active_pixel_indices).astype(np.intp)
#         print(active_pixel_indices)
#         pixel_representation = np.zeros(self.pixel_matrix_dims)
#         pixel_representation[active_pixel_indices] = 1
#         print(pixel_representation)
#         pixel_representation = ndimage.maximum_filter(pixel_representation, size=self.pixel_width_of_a_dot)
#         return normalized_representation, pixel_representation
#
#
# if __name__ == '__main__':
#     # set parameters
#     params = {
#         'num_dots': 2,
#         'num_pixels': 16,
#         'pixel_width_of_a_dot': 1
#     }
