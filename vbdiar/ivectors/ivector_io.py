import numpy as np
import struct
import binascii
import StringIO


def ivector_to_string(ivector, n_data, metadata=None):
    """ Constructs a string of bytes representing the binary version of the
    ivector.

    Args:
        ivector: numpy vector
        n_data: amount of data used for i-vector extraction
        metadata: string containing encoded metadata

    Returns: 
        String of bytes representing the binary i-vector

    The format for integers is unsigned, four-bytes, little endian. The format
    for floats is four-bytes, little endian.

    """

    init_string = "VBS1"
    vbs_version = 1
    ivec_len    = len(ivector)
    mdata_len   = 0 if metadata is None else len(metadata)

    buf  = struct.pack("< 4s i f i", "VBS1", vbs_version, n_data, ivec_len)
    buf += ivector.astype("<f").tostring()

    if metadata is None:
        buf += struct.pack("<i", 0)
    else:
        buf += struct.pack("<i", len(metadata)) + metadata

    crc32 =  binascii.crc32(buf)

    buf += struct.pack("<i", crc32)

    return buf


def string_to_ivector(buf):
    """ Parses the binary buffer and extracts the i-vector, the number of frames

    Args:
        buf: string buffer with packed data

    Returns:
        ivector: numpy vector
        n_data: amount of data used for i-vector extraction
        metadata: string containing encoded metadata

    The format for integers is unsigned, four-bytes, little endian. The format
    for floats is four-bytes, little endian.

    """

    if isinstance(buf, str):
        inp = StringIO.StringIO(buf)
    else:
        inp = buf

    cbuf = inp.read(4)
    crc  = binascii.crc32(cbuf)
    init_string = struct.unpack("4s", cbuf)[0]

    if init_string != "VBS1":
        raise Exception("Probably wrong data")

    cbuf = inp.read(4)
    crc  = binascii.crc32(cbuf,crc)
    vbs_version = struct.unpack("<i", cbuf)[0]

    if vbs_version != 1:
        raise Exception("Wrong version of VBS format")

    cbuf = inp.read(4)
    crc  = binascii.crc32(cbuf,crc)
    n_data   = struct.unpack("<f", cbuf)[0]

    cbuf = inp.read(4)
    crc  = binascii.crc32(cbuf,crc)
    ivec_len = struct.unpack("<i", cbuf)[0]

    if ivec_len < 1:
        raise Exception("i-vector dimensionality is claimed to be less than 1")

    cbuf = inp.read(ivec_len*4)
    crc  = binascii.crc32(cbuf,crc)
    ivector = np.fromstring(cbuf, dtype="<f")

    cbuf = inp.read(4)
    crc  = binascii.crc32(cbuf,crc)
    n_metadata = struct.unpack("<i", cbuf)[0]

    if n_metadata > 0:
        cbuf = inp.read(n_metadata)
        crc  = binascii.crc32(cbuf,crc)
        metadata = cbuf
    else:
        metadata = None

    cbuf = inp.read(4)
    fcrc = struct.unpack("<i", cbuf)[0]

    if fcrc != crc:
        raise Exception("File CRC32 chcecksum does not match the stored value")

    return ivector, n_data, metadata


def write_binary_ivector(fname, ivector, n_data, metadata=None):
    """ Writes the binary i-vector to a file

    Args:
        fname: file name to use
        ivector: numpy vector
        n_data: amount of data used for i-vector extraction
        metadata: string containing encoded metadata

    Returns: 
        Nothing
    
    The format for integers is unsigned, four-bytes, little endian. The format
    for floats is four-bytes, little endian.

    This function calls ivector_to_string to construct the binary string.

    """
    
    buf = ivector_to_string(ivector, n_data, metadata)

    with open(fname, "wb") as f:
        f.write(buf)

        
def read_binary_ivector(fname):
    """ Read binary i-vector from file

    Args:
        fname: file name to use

    Returns: 
        ivector: numpy vector
        n_data: amount of data used for i-vector extraction
        metadata: string containing encoded metadata
    
    The format for integers is unsigned, four-bytes, little endian. The format
    for floats is four-bytes, little endian.

    This function calls string_to_ivector to re-construct data from the binary
    string.

    """
    
    f= open(fname, "rb")
    ivector, n_data, metadata = string_to_ivector(f)

    return ivector, n_data, metadata
