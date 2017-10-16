import numpy as np

from os import SEEK_SET, SEEK_END
from struct import unpack

from .utils import is_sorted, PrettyDuration

class SpikeGadgetsRecFileReader():
    """This class can read and extract data from rec files recorded with
    SpikeGadgets hardware."""

    def __init__(self, *, start_byte_size = None, timestamp_size = None, n_channels = None,
                 bytes_per_neural_channel = None, header_size = None):
        """Initializes SpikeGadgetsRecFileReader class.

        Parameters
        ==========
        start_byte_size : int, optional
            Number of bytes for the start byte of a packet.
            Defaults to 1
        timestamp_size : int, optional
            Number of bytes per timestamp.
            Defaults to 4
        n_channels : int, optional
            Number of channels containing neural data.
            Defaults to 0.
        bytes_per_neural_channel : int, optional
            Defaults to 2.
        header_size : int, optional
            Number of bytes for the header section of a packet
            Defaults to the value of start_byte_size
        """
        # set defaults:
        if start_byte_size is None:
            start_byte_size = 1
        if timestamp_size is None:
            timestamp_size = 4
        if n_channels is None:
            n_channels = 0
        if bytes_per_neural_channel is None:
            bytes_per_neural_channel = 2

        # size of embedded workspace
        self.config_section_size = None
        # minimum size of any packet
        self.start_byte_size = start_byte_size
        self.header_size = self.start_byte_size
        self.timestamp_size = timestamp_size
        # not every recording will have neural data
        self.n_channels = n_channels
        self.n_spike_channels = None
        self.bytes_per_neural_channel = bytes_per_neural_channel
        self.neural_data_size = self.n_channels * self.bytes_per_neural_channel

    def get_timestamp_bounds(self, filename):
        """Returns the first and last timestamps recorded in the .rec file.

        Parameters
        ==========
        filename : string, path to .rec file

        Returns
        =======
        (first_timestamp, last_timestamp) : tuple of the first and last timestamps
            contained in the .rec file
        """
        # need to determine configuration info of file
        if (self.config_section_size is None) or (self.filename != filename):
            self.filename = filename
            self.get_config_info(filename)

        with open(filename, 'rb') as f:
            # find first and last timestamps of file
            f.seek(self.config_section_size, SEEK_SET)
            packet = f.read(self.packet_size)
            # unlikely but could happen in theory
            if (len(packet) < self.packet_size):
                raise ValueError("Insufficient data in first packet: packet size is {} bytes".format(len(packet)))
            timestamp_start = self.header_size
            # <I format - assumes that the timestamp is an uint32
            first_timestamp = unpack('<I', packet[timestamp_start:timestamp_start + self.timestamp_size])[0]
            f.seek(-self.packet_size, SEEK_END)
            packet = f.read(self.packet_size)
            if (len(packet) < self.packet_size):
                raise ValueError("Insufficient data in last packet: packet size is {} bytes".format(len(packet)))
            last_timestamp = unpack('<I', packet[timestamp_start:timestamp_start + self.timestamp_size])[0]

        return (first_timestamp, last_timestamp)

    def get_config_info(self, filename):
        """Parses configuration information defined in the embedded workspace of 
        a .rec file.

        Parameters
        ==========
        filename : string, path to .rec file from which to determine configuration
            information as defined by the embedded workspace

        Returns
        =======
        None
        """
        import xml.etree.ElementTree as ET
        header_size = 1
        xmlstring = None
        self.reindex_arr = []

        # read .rec file embedded workspace and copy to a string
        with open(filename, 'rb') as f:
            instr = f.readline()
            ii = 0
            while(instr != b'</Configuration>\n'):
                instr = f.readline()
                ii += 1
                # infinite loop protection
                if ii > 1000:
                    raise ValueError("Configuration info not found - check input file")
            self.config_section_size = f.tell()
            f.seek(0, SEEK_SET)
            xmlstring = f.read(self.config_section_size)

        # create xml tree from copied embedded workspace string
        xmltree = ET.ElementTree(ET.fromstring(xmlstring))

        root = xmltree.getroot()
        hw_config = root.find("HardwareConfiguration")
        # calculate packet size
        if hw_config is None:
            # no hardware, no data, at least for now
            raise ValueError("No hardware configuration defined!")
        else:
            self.n_channels = int(hw_config.get("numChannels"))
            self.neural_data_size = self.n_channels * self.bytes_per_neural_channel
            for elements in hw_config.getchildren():
                header_size += int(elements.get("numBytes"))
            # Find all elements with tag "SpikeChannel"
            # Note that the order of elements in reindex_arr will be in the
            # same order as document order. For example, if we're reading
            # tetrode data, the first four elements in reindex_arr correspond to
            # the channels of tetrode 1, the next four correspond to tetrode 2, etc.
            for spike_channel in root.iter("SpikeChannel"):
                self.reindex_arr.append(int(spike_channel.get("hwChan")))
            # Faster if we convert the native Python list to a numpy array when we reindex
            self.reindex_arr = np.array(self.reindex_arr)

        self.n_spike_channels = len(self.reindex_arr)
        self.header_size = header_size
        # every packet needs a timestamp
        self.packet_size = self.header_size + self.timestamp_size + self.neural_data_size

    def read_block(self, file, block_size = None):
        """Reads a block of neural data in a .rec file.

        Parameters
        ==========
        file : an open file object to read from.
        block_size: int, optional
            Number of packets to read and return each time. Default is 1024.

        Returns
        =======
        timestamps: list of timestamps (in samples)
        channel_data: ndarray of size (n_channels, n_samples), where n_samples are
        the actual number of read samples, up to a maximum of block_size.
        """

        # need to determine configuration info of file
        if (self.config_section_size is None) or (self.filename != file.name):
            self.filename = file.name
            # if read_block() is called when the file is already opened, the
            # get_config_info() method will open that file again. So get_config_info
            # better open the file in read-only mode!
            self.get_config_info(file.name)
            #raise ValueError("rec file has not been properly intialized yet in SpikeGadgetsRecReader!")

        if block_size is None:
            block_size = 1024

        # set data types used for reading into numpy array
        header_type = np.uint8
        if self.timestamp_size == 4:
            timestamp_type = np.uint32
        else:
            raise ValueError("Unsupported data type for timestamps!")
        if self.bytes_per_neural_channel == 2:
            neural_channel_type = np.int16
        else:
            raise ValueError("Unsupported data type for a neural channel!")

        # file pointer in config/embedded workspace section. Seek to location
        # where neural data starts
        if file.tell() <= self.config_section_size:
            file.seek(self.config_section_size, SEEK_SET)

        if self.n_channels == 0:
            raise ValueError("Expect no neural data to be recorded on rec file!")
        # will only read complete number of packets, so not a problem if the
        # block size is larger than the number of packets we actually read
        dt = np.dtype([('header', header_type, (self.header_size,)),
                       ('timestamps', timestamp_type),
                       ('channel_data', neural_channel_type, (self.n_channels,))])
        data = np.fromfile(file, dtype=dt, count=block_size)
        timestamps = data['timestamps'].tolist()
        # reorder and extract only the channels we want
        channel_data = (data['channel_data'].T)[self.reindex_arr]

        return timestamps, channel_data


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
        start_byte_size : int, optional
        timestamp_size : int, optional
        bytes_per_neural_channel: int, optional
        """
        self.file_list = None
        self._tsamples_starts = []
        self._tsamples_stops = []

        # initialize sampling rate
        fs = kwargs.get('fs', None)
        if fs is None:
            fs = 30000
        self.fs = fs

        kwargs['start_byte_size'] = kwargs.get('start_byte_size', None)
        kwargs['timestamp_size'] = kwargs.get('timestamp_size', None)
        kwargs['bytes_per_neural_channel'] = kwargs.get('bytes_per_neural_channel', None)

        self._reader = SpikeGadgetsRecFileReader(**kwargs)

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
        return "<JagularFileMap: %s>%s" % (nstr, address_str)
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
