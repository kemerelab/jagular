import numpy as np

from os import SEEK_SET, SEEK_END
from struct import unpack
from abc import ABC, abstractmethod

########################################################################
# class JagularFileReader
########################################################################
class JagularFileReader(ABC):
    """Base class for JagularFileReader objects."""

    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def get_timestamp_bounds(self, filename):
        """Returns the first and last timestamps recorded in a particular file.

        Most commonly, the filename would be a .rec file, and the JagularFileMap
        will call get_timestamp_bounds for each file in its list, so as to build
        up a list of timestamp boundaries.

        However, this assumes that the timestamp information is available in
        'filename', which is true for a SpikeGadgets .rec file, but is no longer
        true when we read a .raw channel file, for example, in which case the
        timestamps are contained in a seperate timestamps file. In such a case,
        the 'filename' argument does not really make sense, but it is still here
        for backward compatibility. It could be changed, however, since it is
        not part of the publically-facing API.
        """
        return

    @abstractmethod
    def read_block(self, file, block_size=None):
        """Reads a block of data, and returns the timestamps, and data for the
        block.

        Here, file is an OPEN file pointer, so all we need to implement here
        is to read the next block_size data from the open file. We do not need
        to open the file, and we do not need to seek to any specific place.
        Simply read the next block_size data and return the timestamps and data
        corresponding to that block.

        You also don't need to be careful about checking for boundary conditions
        since all of that is handled for you by JagularFileMap when it consumes
        this method.
        """
        # return timestamps, channel_data
        return


########################################################################
# class SpikeGadgetsSingleChannelReader
########################################################################
class SpikeGadgetsSingleChannelReader(JagularFileReader):
    """This class can read and parse data from a raw channel file extracted from
    SpikeGadgets hardware."""

    def __init__(self, *, required_arg, timestamps_file=None, timestamps_dtype=None, dtype=None):
        """Initializes SpikeGadgetsSingleChannelReader class.

        Parameters
        ==========
        timestamps_file : str, optional
            Filename of timestamps file. If None, then timestamps will be
            inferred, starting from zero, and assuming no breaks in the channel
            data.
        timestamps_dtype : dtype, optional
            Defaults to np.uint64
        dtype : dtype, optional
            Defaults to np.int16
        """
        # set defaults:
        if timestamps_dtype is None:
            timestamps_dtype = np.uint64
        if dtype is None:
            dtype = np.int16

        self.timestamps_file = timestamps_file
        self.timestamps_dtype = timestamps_dtype
        self.dtype = dtype # channel data dtype

    def get_timestamp_bounds(self, filename):
        """Returns the first and last timestamps recorded in the timestamps file.

        Parameters
        ==========
        filename : string, path to .raw file (NOT TIMESTAMPS FILE!)

        Returns
        =======
        (first_timestamp, last_timestamp) : tuple of the first and last timestamps
            contained in the timestamps file
        """
        # return 0, 5
        raise NotImplementedError

    def read_block(self, ch_file, block_size=None):
        """Reads a block of neural data from a .raw channel file.

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

        if block_size is None:
            block_size = 1024

        channel_data = np.fromfile(ch_file, dtype=self.dtype, count=block_size)
        if self.timestamps_file is None:
            raise NotImplementedError('we do not yet support auto timestamp inderence')
        else:
            timestamps = np.fromfile(self.timestamps_file, dtype=self.timestamps_dtype, count=block_size)

        return timestamps, channel_data


########################################################################
# class SpikeGadgetsRecFileReader
########################################################################
class SpikeGadgetsRecFileReader(JagularFileReader):
    """This class can read and extract data from rec files recorded with
    SpikeGadgets hardware."""

    def __init__(self, *, start_byte_size=None, timestamp_size=None, n_channels=None,
                 bytes_per_neural_channel=None, header_size=None):
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
        self.reindex_arr = np.array([])
        unconverted_hw_chan_list = []

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
                unconverted_hw_chan_list.append(int(spike_channel.get("hwChan")))
            n_cards, rem = divmod(self.n_channels, 32)
            if rem != 0:
                raise ValueError("Number of neural channels must be a multiple of 32")
            # Convert hw channels defined in workspace to actual hardware channel. The
            # actual hardware channel tells us the packet location of the desired data
            unconverted_hw_chan_arr = np.array(unconverted_hw_chan_list)
            self.reindex_arr = (((unconverted_hw_chan_arr % 32) * n_cards) 
                               + np.floor(unconverted_hw_chan_arr / 32))
            #print(unconverted_hw_chan_list)
            self.reindex_arr = self.reindex_arr.astype(int)
            #print(self.reindex_arr)
            #print(self.reindex_arr.dtype)

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