import numpy as np

from .utils import is_sorted, PrettyDuration
from .readers import SpikeGadgetsRecFileReader

class JagularFileMap(object):
    """Helper class to read from multiple files spanning a conceptually continuous segment of data."""

    def __init__(self, *files, **kwargs):
        """Docstring goes here!

        Parameters
        ==========
        files : tuple, list, or multiple unnamed arguments, optional
            Collectin of filenames to add to JagularFileMap.
        fs : float, optional
            Sampling rate (in Hz). Default value is 30,000 Hz.
        reader : JagularFileReader, optional
            JagularFileReader object; default is SpikeGadgetsRecFileReader
            - start_byte_size : int, optional
            - timestamp_size : int, optional
            - bytes_per_neural_channel: int, optional
        """
        self.file_list = None
        self._tsamples_starts = []
        self._tsamples_stops = []

        # initialize sampling rate
        fs = kwargs.get('fs', None)
        if fs is None:
            fs = 30000
        self.fs = fs

        reader = kwargs.get('reader', None)
        kwargs.pop('reader', None)
        if reader is None:
            reader = SpikeGadgetsRecFileReader
            kwargs['start_byte_size'] = kwargs.get('start_byte_size', None)
            kwargs['timestamp_size'] = kwargs.get('timestamp_size', None)
            kwargs['bytes_per_neural_channel'] = kwargs.get('bytes_per_neural_channel', None)

        self._reader = reader(**kwargs)

        file_tuple = self._get_file_tuple(files)
        if file_tuple:
            self.add_files(file_tuple)

    def _get_file_tuple(self, files):
        """'files' can be a list, tuple, or multiple arguments; returns a tuple."""
        try:
            if isinstance(files[0], (tuple, list)):
                files = files[0]
            return tuple(files)
        except IndexError:
            return ()

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<Empty JagularFileMap>%s" % (address_str)
        elif self.n_files > 1:
            nstr = "{} files spanning {} (missing {} between files)".format(self.n_files, self.duration_w_gaps, self._inter_gap_duration())
        else:
            nstr = "1 file spanning {}".format(self.duration_w_gaps)
        return "<JagularFileMap[%s]: %s>%s" % (str(self._reader), nstr, address_str)
        #TODO: want something like this: <JagularFileMap: 5 files spanning 23:45:17 hours (missing 23:46 minutes)> at 0x2a039e201d0

    def add_files(self, *files):
        """Add files to internal list, and populate time boundaries."""
        # here we will read the files to extract first and last timestamps,
        # and we will update (not replace!) the internal list of files and
        # timestamps

        #TODO: check if value already exists, and warn user and take appropriate action
        file_tuple = self._get_file_tuple(files)

        for file in file_tuple:
            first_timestamp, last_timestamp = self._reader.get_timestamp_bounds(file)
            assert first_timestamp <= last_timestamp, "first_timestamp > last_timestamp for file '{}'! Aborting...".format(file)
            self._tsamples_starts.append(first_timestamp)
            self._tsamples_stops.append(last_timestamp)
            if self.file_list:
                self.file_list.append(file)
            else:
                self.file_list = []
                self.file_list.append(file)

        if not self.issorted:
            self._sort()

    def _sort(self):
        """Sort filenames, and timestamps according to starting timestamps."""
        new_order = sorted(range(len(self._tsamples_starts)), key=lambda k: self._tsamples_starts[k])
        self._tsamples_starts = np.array(self._tsamples_starts)[new_order].tolist()
        self._tsamples_stops = np.array(self._tsamples_stops)[new_order].tolist()
        self.file_list = np.array(self.file_list)[new_order].tolist()

    def _inter_gap_duration(self):
        """Total duration of gaps (in seconds) missing between files."""
        return PrettyDuration((self.timesamples[1:,0] - self.timesamples[:-1,1]).sum()/self.fs)

    @property
    def timestamps(self):
        """Timestamps (in seconds) array with size (n_files, 2), with each row as (start, stop)."""
        return np.vstack((self._tsamples_starts, self._tsamples_stops)).T / self.fs

    @property
    def timesamples(self):
        """Timestamps (in samples) array with size (n_files, 2), with each row as (start, stop)."""
        return np.vstack((self._tsamples_starts, self._tsamples_stops)).T

    @property
    def issorted(self):
        """Returns True if timestamps are monotonically increasing."""
        return is_sorted(self._tsamples_starts)

    @property
    def isempty(self):
        """(bool) Empty JagularFileMap."""
        try:
            if len(self.file_list) > 0:
                return False
            else:
                return True
        except TypeError: # file_list is None
            return True

    @property
    def start(self):
        """First timestamp (in samples) in JagularFileMap."""
        if not self.isempty:
            return self.timesamples[0,0]
        else:
            return np.inf

    @property
    def stop(self):
        """Last timestamp (in samples) in JagularFileMap."""
        if not self.isempty:
            return self.timesamples[-1,1]
        else:
            return -np.inf

    @property
    def start_time(self):
        """First timestamp (in seconds) in JagularFileMap."""
        if not self.isempty:
            return self.timestamps[0,0]
        else:
            return np.inf

    @property
    def stop_time(self):
        """Last timestamp (in seconds) in JagularFileMap."""
        if not self.isempty:
            return self.timestamps[-1,1]
        else:
            return -np.inf

    @property
    def duration_w_gaps(self):
        """Total duration (in seconds) mapped by file objects, including potential gaps."""
        if self.isempty:
            return PrettyDuration(0)
        else:
            return PrettyDuration((self.stop_time - self.start_time))

    @property
    def duration_wo_gaps(self):
        """Total duration (in seconds) mapped by file objects, excluding inter-file gaps.
        NOTE: intra-file gaps are not taken into account here, but should be relatively small.
        """
        if self.isempty:
            return PrettyDuration(0)
        else:
            return PrettyDuration(np.diff(self.timestamps).sum())

    @property
    def durations(self):
        """Durations (in seconds) for each file object."""
        if self.isempty:
            return PrettyDuration(0)
        elif self.n_files == 1:
            return PrettyDuration(np.diff(self.timestamps).squeeze())
        else:
            return [PrettyDuration(duration) for duration in np.diff(self.timestamps).squeeze()]

    @property
    def n_files(self):
        """Number of files in JagularFileMap."""
        if self.isempty:
            return 0
        return len(self.file_list)

    def plot(self):
        """Plot the times spanned by all the files contained in the JagularFileMap."""
        from nelpy.core import EpochArray
        from nelpy.plotting import epochplot

        ax = epochplot(EpochArray(self.timestamps))

        return ax

    def _samples_within_bounds(self, start, stop):
        """Check that [start, stop] is fully contained (inclusive) of [self.start, self.stop]"""
        if stop < start:
            raise ValueError("start time has to be less or equal to stop time!")
        if start < self.start:
            raise ValueError("requested start time is earlier than first avaialbe timestamp (={})!".format(self.start))
        if stop > self.stop:
            raise ValueError("requested stop time is later than last avaialbe timestamp (={})!".format(self.stop))
        return True

    def _time_within_bounds(self, start, stop):
        """Check that [start, stop] is fully contained (inclusive) of [self.start_time, self.stop_time]"""
        if stop < start:
            raise ValueError("start time has to be less or equal to stop time!")
        if start < self.start_time:
            raise ValueError("requested start time is earlier than first avaialbe timestamp (={})!".format(self.start_time))
        if stop > self.stop_time:
            raise ValueError("requested stop time is later than last avaialbe timestamp (={})!".format(self.stop_time))
        return True

    def request_data(self, start, stop, interpolate=True):
        """Return data between start and stop (in seconds?), inclusive."""

        # check that request is within allowable bounds
        if self._time_within_bounds(start, stop):
            # invoke reader here
            pass

        raise NotImplementedError

    def read_stitched_files(self, block_size=None):
        """Yield all data one block at a time, from multiple files that are stitched together.

        Parameters
        ==========
        block_size: int, optional
            Number of packets to read in each block. Default is 1024.

        Returns
        =======
        block of packets at a time, as a tuple (timestamps, channel_data)
        """

        from contextlib import ExitStack

        if self.isempty:
            raise ValueError("Cannot read data from an empty JagularFileMap.")

        if block_size is None:
            block_size = 1024 # number of samples to read per step

        with ExitStack() as stack:
            files = [stack.enter_context(open(fname, 'rb')) for fname in self.file_list]
            ii=0
            while True:
                try:
                    timestamps, channel_data = self._reader.read_block(file=files[ii], block_size=block_size)
                    while 0 < len(timestamps) < block_size:
                        # block_size could not be filled from current file, so advance to next file
                        ii+=1
                        timestamps_, channel_data_ = self._reader.read_block(file=files[ii], block_size=block_size-len(timestamps))
                        if not isinstance(timestamps, list):
                            raise TypeError("timestamps MUST be a list!")
                        timestamps = timestamps + timestamps_ # list concatenation
                        if channel_data_ is not None:
                            channel_data = np.hstack((channel_data, channel_data_))
                    if timestamps:
                        yield timestamps, channel_data
                    else:
                        ii+=1
                except IndexError:
                    # no more files are available, but we may still have some non-yielded data
                    if timestamps:
                        yield timestamps, channel_data
                        return
                    else:
                        return

















        # FB! Check this out: http://neopythonic.blogspot.com/2008/10/sorting-million-32-bit-integers-in-2mb.html
        # Also check http://effbot.org/zone/wide-finder.htm for multiprocessing etc., but not sure about closing files?
        # and here: for multiple ctx managers: https://stackoverflow.com/questions/3024925/python-create-a-with-block-on-several-context-managers

        # def getData(filename1, filename2):
        #     with open(filename1, "rb") as csv1, open(filename2, "rb") as csv2:
        #         reader1 = csv.reader(csv1)
        #         reader2 = csv.reader(csv2)
        #         for row1, row2 in zip(reader1, reader2):
        #             yield (np.array(row1, dtype=np.float),
        #                 np.array(row2, dtype=np.float))
        #                 # This will give arrays of floats, for other types change dtype

        # for tup in getData("file1", "file2"):
        #     print(tup)


        ####################

        # def read_file(path, block_size=1024):
        #     with open(path, 'rb') as f:
        #         while True:
        #             piece = f.read(block_size)
        #             if piece:
        #                 yield piece
        #             else:
        #                 return

        # for piece in read_file(path):
        #     process_piece(piece)

        # # pseudo code (used outside of JagularFileMap):
        # with open(filename, 'wb') as fout:
        #     for piece in jfm.read_all():
        #         filtered = scipy.filter(piece)
        #         fout.save(filtered)
