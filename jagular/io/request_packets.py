import os
import xml.etree.ElementTree as ET
import struct

class FileInfo():
    #START_BYTE_SIZE = 1
    #TIMESTAMP_SIZE = 4
    #BYTES_PER_NEURAL_CHANNEL = 2
 
    def __init__(self, filename):
        self.start_byte_size = 1
        self.timestamp_size = 4
        self.bytes_per_neural_channel = 2
        # minimum size of any packet
        self.header_size = self.start_byte_size
        self.filename = filename

    def get_timestamp_bounds(self, filename):
        # read .rec file embedded workspace and copy to a string
        with open(filename, 'rb') as infile:
            instr = infile.readline()
            while(instr != b'</Configuration>\n'):
                instr = infile.readline()
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
        
        return (first_timestamp, last_timestamp, infile)        
