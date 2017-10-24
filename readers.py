class SpikeGadgetsSingleChannelReader():
    """This class can read and parse data from a raw channel file extracted from
    SpikeGadgets hardware."""

    def __init__(self, *, timestamp_size=None, dtype=None):
        """Initializes SpikeGadgetsRecFileReader class.

        Parameters
        ==========
        timestamp_size : int, optional
            Number of bytes per timestamp.
            Defaults to 4
        bytes_per_sample : int, optional
            Defaults to 2.
        header_size : int, optional
            Number of bytes for the header section of a packet
            Defaults to the value of start_byte_size
        """
        # set defaults:
        if timestamp_size is None:
            timestamp_size = 4
        if bytes_per_sample is None:
            bytes_per_sample = 2

        self.timestamp_size = timestamp_size
        self.bytes_per_sample = bytes_per_sample

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