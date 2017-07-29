import numpy as np

class SpikeGadgetsRecFileReader():

    def __init__(self, *, start_byte_size=None, timestamp_size=None, bytes_per_neural_channel=None, header_size=None):
        # set defaults:
        if start_byte_size is None:
            start_byte_size = 1
        if timestamp_size is None:
            timestamp_size = 4
        if bytes_per_neural_channel is None:
            bytes_per_neural_channel = 2
        if header_size is None:
            header_size = start_byte_size

        self.start_byte_size = start_byte_size
        self.timestamp_size = timestamp_size
        self.bytes_per_neural_channel = bytes_per_neural_channel
        # minimum size of any packet
        self.header_size = header_size

    def get_timestamp_bounds(self, filename):
        import os
        import struct
        import xml.etree.ElementTree as ET

        #TODO: JOSH LET'S FIX THIS!!!
        self.header_size = 1
        ii = 0
        # read .rec file embedded workspace and copy to a string
        with open(filename, 'rb') as infile:
            instr = infile.readline()
            while(instr != b'</Configuration>\n'):
                instr = infile.readline()
                ii += 1
                # infinite loop protection
                if ii > 1000:
                    print("Configuration info not found - check input file")
                    break
            config_section_size = infile.tell()
            infile.seek(0, os.SEEK_SET)
            xmlstring = infile.read(config_section_size)

            # create xml tree from copied embedded workspace string
            tree = ET.ElementTree(ET.fromstring(xmlstring))
            root = tree.getroot()
            hw_config = root.find("HardwareConfiguration")

            # calculate packet size
            if hw_config is None:
                print("No hardware configuration defined!")
            self.neural_data_size = int(hw_config.get("numChannels"))*self.bytes_per_neural_channel
            for elements in hw_config.getchildren():
                self.header_size += int(elements.get("numBytes"))

            # every packet needs a timestamp
            self.packet_size = self.header_size + self.timestamp_size + self.neural_data_size

            # find first and last timestamps of file
            infile.seek(config_section_size, os.SEEK_SET)
            packet = infile.read(self.packet_size)
            timestamp_start = self.header_size
            # <I format - assumes that the timestamp is an uint32
            first_timestamp = struct.unpack('<I', packet[timestamp_start:timestamp_start + self.timestamp_size])[0]
            infile.seek(-self.packet_size, os.SEEK_END)
            packet = infile.read(self.packet_size)
            last_timestamp = struct.unpack('<I', packet[timestamp_start:timestamp_start + self.timestamp_size])[0]

        return (first_timestamp, last_timestamp, filename)


class JagularFileMap(object):
    """Helper class to read from multiple files spanning a conceptually continuous segment of data."""

    def __init__(self, *files, **kwargs):
        self.file_list = None
        self._ts_starts = []
        self._ts_stops = []

        kwargs['start_byte_size'] = kwargs.get('start_byte_size', None)
        kwargs['timestamp_size'] = kwargs.get('timestamp_size', None)
        kwargs['bytes_per_neural_channel'] = kwargs.get('bytes_per_neural_channel', None)
        kwargs['header_size'] = kwargs.get('header_size', None)

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
        return "<JagularFileMap>%s" % (address_str)

    def __pairwise(self, iterable):
        """returns a zip of all neighboring pairs.
        This is used as a helper function for is_sorted.

        Example
        -------
        >>> mylist = [2, 3, 6, 8, 7]
        >>> list(pairwise(mylist))
        [(2, 3), (3, 6), (6, 8), (8, 7)]
        """
        from itertools import tee

        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def __is_sorted(self, iterable, key=lambda a, b: a <= b):
        """Check to see if iterable is monotonic increasing (sorted)."""
        return all(key(a, b) for a, b in self.__pairwise(iterable))

    def add_files(self, *files):
        """Add files to internal list, and populate time boundaries."""
        # here we will read the files to extract first and last timestamps,
        # and we will update (not replace!) the internal list of files and
        # timestamps

        #TODO: check if value already exists, and warn user and take appropriate action
        file_tuple = self._get_file_tuple(files)

        for file in file_tuple:
            first_timestamp, last_timestamp, infile = self._reader.get_timestamp_bounds(file)
            assert first_timestamp <= last_timestamp, "first_timestamp > last_timestamp for file '{}'! Aborting...".format(file)
            self._ts_starts.append(first_timestamp)
            self._ts_stops.append(last_timestamp)
            if self.file_list:
                self.file_list.append(file)
            else:
                self.file_list = []
                self.file_list.append(file)

        if not self.issorted:
            self._sort()

    def _sort(self):
        """Sort filenames, and timestamps according to starting timestamps."""
        new_order = sorted(range(len(self._ts_starts)), key=lambda k: self._ts_starts[k])
        self._ts_starts = np.array(self._ts_starts)[new_order].tolist()
        self._ts_stops = np.array(self._ts_stops)[new_order].tolist()
        self.file_list = np.array(self.file_list)[new_order].tolist()

    @property
    def timestamps(self):
        """Timestamps array with size (n_files, 2), with each row as (start, stop)."""
        return np.vstack((self._ts_starts, self._ts_stops)).T

    @property
    def issorted(self):
        """Returns True if timestamps are monotonically increasing."""
        return self.__is_sorted(self._ts_starts)

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
        """First timestamp in JagularFileMap."""
        if not self.isempty:
            return self.timestamps[0,0]
        else:
            return np.inf

    @property
    def stop(self):
        """Last timestamp in JagularFileMap."""
        if not self.isempty:
            return self.timestamps[-1,1]
        else:
            return -np.inf

    @property
    def duration_w_gaps(self):
        """Total duration (in number of samples) mapped by file objects, including potential gaps."""
        if self.isempty:
            return 0
        else:
            return self.stop - self.start

    @property
    def duration_wo_gaps(self):
        """Total duration (in number of samples) mapped by file objects, excluding inter-file gaps.
        NOTE: intra-file gaps are not taken into account here, but should be relatively small.
        """
        raise NotImplementedError("not yet implemebted!")

    @property
    def n_files(self):
        """Number of files in JagularFileMap."""
        if self.isempty:
            return 0
        return len(self.file_list)

    def _within_bounds(self, start, stop):
        """Check that [start, stop] is fully contained (inclusive) of [self.start, self.stop]"""
        if stop < start:
            raise ValueError("start time has to be less or equal to stop time!")
        if start < self.start:
            raise ValueError("requested start time is earlier than first avaialbe timestamp (={})!".format(self.start))
        if stop > self.stop:
            raise ValueError("requested stop time is later than last avaialbe timestamp (={})!".format(self.stop))
        return True

    def request_data(self, start, stop, interpolate=True):
        """Return data between start and stop, inclusive."""

        # check that request is within allowable bounds
        if self._within_bounds(start, stop):
            # invoke reader here
            pass

        raise NotImplementedError

    def read_all(self, block_size=None, pad_before=None, pad_after=None, interpolate=True):
        """Yield all data one block at a time."""

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

        raise NotImplementedError




        