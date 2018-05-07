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

# import numpy as np
# from os.path import isfile, getsize
# from os import SEEK_SET
# from contextlib import ExitStack
# from jagular.utils import PrettyDuration
# from time import time, strfrtime, localtime
# class SpikeGadgetsRecFileReader(object):
#     """This class can read and extract data from .rec files recorded with
#     SpikeGadgets hardware."""

#     def __init__(self, filepath, *, fs=None, start_type=None, 
#                  timestamp_type=None, single_neural_channel_type=None):
#         """Initializes SpikeGadgetsRecFileReader class.
#         """
#         # set defaults:
#         if fs is None:
#             fs = 30000
#         if start_type is None:
#             start_type = '<u1'
#         if timestamp_type is None:
#             timestamp_type = '<u4'
#         if single_neural_channel_type is None:
#             single_neural_channel_type = '<i2'

#         self._filepath = filepath
#         self._fs = fs
#         self._start_type = start_type
#         self._timestamp_type = timestamp_type
#         self._single_neural_channel_type = single_neural_channel_type
#         self._dtype = None
#         self._reindex_arr = None

#         if not isfile(self._filepath):
#             raise IOError("%s does not exist", self._filepath)
        
#         self._parse_file()
#         self._calc_time_info()
        
#     def __repr__(self):
#         # TODO: Display total time and gaps
#         address_str = " at " + str(hex(id(self)))

#         return "<SpikeGadgetsRecFileReader[%s]: >%s" % (str(self._filepath), address_str)    

#     def _parse_file(self, *, maxlines=None):
#         """Parses configuration section of rec file and generates dtype that can be
#         used to extract data from the file

#         Parameters
#         ==========
#         maxlines : Maximum number of lines to read from the start of the file
#             to determine configuration info. This value serves as infinite
#             loop protection in case the file has an invalid configuration
#             configuration section (aka the embedded workspace)
        
#         """
        
#         # TODO: Figure out better way to parse hardware listed in the
#         # rec file configuration section, and write documentation
#         # detailing any assumptions made
#         if maxlines is None:
#             maxlines = 1000

#         with open(self._filepath, 'rb') as rec_fobj:
#             fileline = []
#             ii = 0
#             while fileline != b'</Configuration>\n':
#                 fileline = rec_fobj.readline()
#                 if ii > maxlines:
#                     raise ValueError("Could not determine configuration info, check rec file")
#                 ii += 1
#                 print(fileline)
#             self._config_size = rec_fobj.tell()
#             # Found end of configuration info, now copy it and parse
#             rec_fobj.seek(0, SEEK_SET)
#             xmlstring = rec_fobj.read(self.config_size)
#             xmltree = ET.ElementTree(ET.fromstring(xmlstring))
#             root = xmltree.getroot()
#             global_config = root.find("GlobalConfiguration")
#             # Don't know yet how to index into correct packet location of 
#             # channel data if saveDisplayedChanOnly is not 0
#             save_displayed_chan_only = int(global_config.get("saveDisplayedChanOnly"))
#             if save_displayed_chan_only != 0:
#                 raise NotImplementedError("Data extraction with saveDisplayedChanOnly = \"%d\"
#                                            not currently supported", save_displayed_chan_only )
#             # These attributes may not have always existed in rec files so handle
#             # errors just in case
#             timestamp_start_local = global_config.get("timestampAtCreation"))
#             timestamp_start_system = global_config.get("systemTimeAtCreation")
#             if timestamp_start_local is None:
#                 warn("No timestampAtCreation attribute defined in rec file")
#             else:
#                 self._timestamp_start_local = np.int64(timestamp_start_local)
#             if timestamp_start_system is None:
#                 warn("No attribute 'systemTimeAtCreation' defined in rec file")
#             else:
#                 self._timestamp_start_system = np.int64(timestamp_start_system)

#             device_dict = {}
#             device_list = []
#             device_order = []
#             # devices such as MCU, ECU, etc.
#             hw_config_element = root.find("HardwareConfiguration")
#             self._n_channels = int(hw_config_element.get('numChannels'))
#             if n_neural_channels == 0:
#                 print("Rec file has no neural channels defined")
#             for device_element in hw_config_element.getchildren():
#                 # The packetOrderPreference attribute describes the location in which a
#                 # device's data will appear in the packet. The lower this attribute's 
#                 # value, the EARLIER it will appear. We use this information to correctly
#                 # construct the numpy data type used to read the rec file. We do this
#                 # because in the workspace (which is then embedded in the rec file), a
#                 # user might not enumerate devices in the same order that their data
#                 # appear in the packet. However, this is poor organization, so if you
#                 # do this, stop it!
#                 device_order.append(int(device_element.get('packetOrderPreference')))
#                 device_list.append(  (device_element.get('name'), int(device_element.get('numBytes'))) )
#                 device_name = device_element.get('name')

#                 device_dict[device_name] = {}
#                 for channel_element in device_element.getchildren():
#                     if (device_name == 'RF') and (channel_element.get('dataType') != 'uint32'):
#                         raise ValueError("RF data type not defined correctly! Must be uint32")
#                     channel_id = channel_element.get('id')
#                     device_dict[device_name][channel_id] = {}
#                     #print(subelement.get('startByte'))
#                     device_dict[device_name][channel_id] = {'start_byte': int(channel_element.get('startByte')),
#                                                             'bit': int(channel_element.get('bit')) }

#             device_list = [device_list[idx] for idx in np.argsort(device_order)]
#             rec_dtype_list = [('start', self.start_type)]
#             for (device_name, num_bytes) in device_list:
#                 rec_dtype_list.append((device_name, '<u%d' % num_bytes))
#             rec_dtype_list.append(('timestamps', self.timestamp_type))
#             rec_dtype_list.append(('channel_data', self.single_neural_channel_type, (self.n_channels,)))
            
#             # Find all elements with tag "SpikeChannel"
#             # Note that the order of elements in reindex_arr will be in the
#             # same order as document order. For example, if we're reading
#             # tetrode data, the first four elements in reindex_arr correspond to
#             # the channels of tetrode 1, the next four correspond to tetrode 2, etc.
#             for spike_channel in root.iter("SpikeChannel"):
#                 unconverted_hw_chan_list.append(int(spike_channel.get("hwChan")))
#             n_cards, rem = divmod(self.n_channels, 32)
#             if rem != 0:
#                 raise ValueError("Number of neural channels must be a multiple of 32")
#             # Convert hw channels defined in workspace to actual hardware channel. The
#             # actual hardware channel tells us the packet location of the desired data
#             unconverted_hw_chan_arr = np.array(unconverted_hw_chan_list)
#             self._reindex_arr = (((unconverted_hw_chan_arr % 32) * n_cards) 
#                                  + np.floor(unconverted_hw_chan_arr / 32))

#             self._reindex_arr = self._reindex_arr.astype(int)
            
#             # Everything went successfully, ok to generate the dtype now
#             self._dtype = np.dtype(rec_dtype_list)
        
#     def _calc_time_info():
#         """Calculates time info of the rec file, first and last timestamps, total timespan,
#         total gaps, etc. All these values are stored as int64 with the same time base as that 
#         of the sampling rate"""
        
#         data = np.memmap(self._filename, dtype=self._dtype, mode='r', offset=self._config_size)
        
#         start = data['timestamps'][0]
#         stop = data['timestamps'][-1]
#         self._timestamp_bounds = np.array([[start, stop]]).astype('<i8')

#         start_time = start - self._timestamp_start_local + np.round(self._timestamp_start_system * self._fs / 1000.0)
#         stop_time  = stop  - self._timestamp_start_local + np.round(self._timestamp_start_system * self._fs / 1000.0)
#         self._timestamp_bounds_absolute = np.array([[start_time, stop_time]]).astype('<i8')
        
#         self._n_packets, rem = divmod(getsize(self._filepath) - self._config_size, self._dtype.itemsize)
#         if rem:
#             raise ValueError("Non-integral number of packets, or the dtype might be wrong")
            
#         self._timespan         = self._timestamp_bounds[-1] - self._timestamp_bounds[0]
#         self._timespan_wo_gaps = self._timespan - self._n_packets
        
#     @property
#     def timestamps():
#         """A (1, 2) numpy array containing the first and
#         last timestamps in the file. The values in
#         the returned array can NOT be converted into a
#         real date and time unambiguously. Units are in
#         the same time base as that of the sampling rate."""
        
#         return self._timestamp_bounds
        
#     @property
#     def timestamps_absolute():
#         """A (1, 2) numpy array containing the first and
#         last timestamps (absolute) in the file. This means
#         that the values in the returned array can be
#         converted into a real date and time unambiguously.
#         Units are in the same time base as that of the
#         sampling rate."""
        
#         return self._timestamp_bounds_absolute
    
#     @property
#     def start(self):
#         """First timestamp recorded in rec file. The
#         returned value can NOT be converted into a real
#         date and time unambiguously. Units are in the
#         same time base as that of the sampling rate."""
#         return self._timestamp_bounds[0]
        
#     @property
#     def stop(self):      
#         """Last timestamp recorded in rec file. The
#         returned value can NOT be converted into a real
#         date and time unambiguously. Units are the same
#         time base as that of the sampling rate."""
#         return self._timestamp_bounds[-1]
        
#     @property
#     def start_absolute(self):
#         """First timestamp (absolute) recorded in rec file.
#         This means that the returned value can be converted
#         into a real date and time unambiguously. Units are
#         in the same time base as that of the sampling rate."""
#         return self._timestamp_bounds_absolute[0]
        
#     @property
#     def stop_absolute(self):       
#         """Last timestamp (absolute) recorded in rec file.
#         This means that the returned value can be converted
#         into a real date and time unambiguously. Units are
#         in the same time base as that of the sampling rate."""
#         return self._timestamp_bounds_absolute[-1]
    
#     @property
#     def start_time(self):      
#         """Human-readable string displaying date and time
#         of the first timestamp recorded in the rec file."""
#         return strfrtime("%a, %m/%d/%Y at %H:%M:%S", 
#                          localtime(self._timestamp_bounds_absolute[0] / self._fs)
    
#     @property
#     def stop_time(self):      
#         """Human-readable string displaying date and time
#         of the last timestamp recorded in the rec file."""
#         return strfrtime("%a, %m/%d/%Y at %H:%M:%S", 
#                          localtime(self._timestamp_bounds_absolute[-1] / self._fs)
    
#     @property
#     def duration_w_gaps():
#         """The amount of time spanned by the data recorded
#         in the rec file, including gaps in the data."""
        
#         return PrettyDuration(self._timespan / self._fs)
        
#     @property
#     def duration_wo_gaps():
#         """The amount of time spanned by the data recorded
#         in the rec file, excluding gaps in the data."""
        
#         return PrettyDuration(self._timespan_wo_gaps / self._fs)
    
#     @property    
#     def filepath(self):
#         """The path to the file associated with this particular
#         SpikeGadgetsRecFileReader instance."""
        
#         return self._filepath
    
#     @filepath.setter
#     def filepath(self, filepath, *, **kwargs):
#         """Changes the file to associate with this particular
#         SpikeGadgetsRecFileReader instance."""
        
#         self.__init__(filepath, **kwargs)
    
#     @property
#     def dtype(self):
#         """The numpy dtype used to read the rec file"""
        
#         return self._dtype
    
#     @dtype.setter
#     def dtype(self, dtype):
#         """Manually set dtype used to read the rec file. Use at
#         your own risk!"""
        
#         self._dtype = dtype
        
#     @property
#     def fs(self):
#         """Sampling rate of the rec file"""
        
#         return self._fs
    
#     @fs.setter
#     def fs(self, val):
#         """Sets sampling rate of the rec file"""
                         
#         self._fs = val
        
#     @property
#     def timestamp(self):
#         """Shows date and time when the rec file was created. 
#         Date format is 'day, mm/dd/yyyy at HH:MM:SS'"""
#         # system time in rec file saved in units of msec
#         return "File created on " + strfrtime("%a, %m/%d/%Y at %H:%M:%S", 
#                                               localtime(self._timestamp_start_system / 1000.0))
    
#     def extract_channels(self, *, block_size=None, subset='all',
#                          ts_out=None, ch_out_prefix=None):
#         """Extracts neural channel data and timestamps"""
        
#         if block_size is None:
#             block_size = 65536
#         if ts_out is None:
#             ts_out = 'timestamps.raw'
#         if subset == 'all':
#             subset = range(self._n_neural_channels)
                         
#         n_chan_zfill = len(str(self._n_neural_channels))

#         ch_out_files = ['ch.' + str(n).zfill(n_chan_zfill) + '.raw' for n in range(self._n_neural_channels)]
        
#         with ExitStack as stack:
#             rec_fobj = stack.enter_context(open(self._filepath, 'rb'))
#             ts_fobj  = stack.enter_context(open(ts_out, 'wb+'))
#             ch_fobjs = [stack.enter_context(open(fname, 'wb+')) for fname in ch_out_files]
#             recfobj.seek(self._config_size, SEEK_SET)
#             data =  self.read_block(recfobj, block_size)
#             channel_data = (data['channel_data'].T)[self._reindex_arr]
                         
#             ts_fojb.write(bytes(data['timestamps']))
#             for ch in range(self._n_neural_channels):
#                 ch_fobjs[ch].write(bytes(channel_data[ch, :]))

#     def extract_imu(self, *, block_size=None):
        
#         """Extracts IMU data from rec file, if it exists"""
        
#     def extract_dio(self, *, block_size=None):
        
#         """Extracts DIOs from rec file, if it exists"""
        
#     def read_block(self, fobj, block_size):
#         """Reads block of data from rec file"""
        
#         return np.fromfile(fobj, dtype=self.dtype, count=block_size)