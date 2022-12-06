import sys
import numpy as np

from ctypes import *
from pysensel.register_map import *
import platform

SENSEL_MAX_DEVICES  = 16

FRAME_CONTENT_PRESSURE_MASK = 0x01
FRAME_CONTENT_LABELS_MASK   = 0x02
FRAME_CONTENT_CONTACTS_MASK = 0x04
FRAME_CONTENT_ACCEL_MASK    = 0x08

CONTACT_MASK_ELLIPSE        =   0x01
CONTACT_MASK_DELTAS         =   0x02
CONTACT_MASK_BOUNDING_BOX   =   0x04
CONTACT_MASK_PEAK           =   0x08

CONTACT_INVALID = 0
CONTACT_START   = 1
CONTACT_MOVE    = 2
CONTACT_END     = 3

class SenselSensorInfo(Structure):
    _fields_ = [("max_contacts", c_ubyte), 
                ("num_rows", c_ushort), 
                ("num_cols", c_ushort), 
                ("width", c_float), 
                ("height", c_float)] 

class SenselContact(Structure):
    _fields_ = [("content_bit_mask", c_ubyte), 
                ("id", c_ubyte), 
                ("state", c_int), 
                ("x_pos", c_float), 
                ("y_pos", c_float), 
                ("total_force", c_float), 
                ("area", c_float),
                ("orientation", c_float), 
                ("major_axis", c_float), 
                ("minor_axis", c_float), 
                ("delta_x", c_float),
                ("delta_y", c_float), 
                ("delta_force", c_float), 
                ("delta_area", c_float), 
                ("min_x", c_float),
                ("min_y", c_float), 
                ("max_x", c_float), 
                ("max_y", c_float),
                ("peak_x", c_float), 
                ("peak_y", c_float), 
                ("peak_force", c_float)] 

class SenselAccelData(Structure):
    _fields_ = [("x", c_int), 
                ("y", c_int), 
                ("z", c_int)] 

class SenselFrameData(Structure):
    _fields_ = [("content_bit_mask", c_ubyte),
                ("lost_frame_count", c_int), 
                ("n_contacts", c_ubyte), 
                ("contacts", POINTER(SenselContact)),
                ("force_array", POINTER(c_float)),
                ("labels_array", POINTER(c_ubyte)),
                ("accel_data", POINTER(SenselAccelData))]

class SenselDeviceID(Structure):
    _fields_ = [("idx", c_ubyte), 
                ("serial_num", c_ubyte*64), 
                ("com_port", c_ubyte*64)] 

class SenselDeviceList(Structure):
    _fields_ = [("num_devices", c_ubyte), 
                ("devices", SenselDeviceID*SENSEL_MAX_DEVICES)] 

class PySensel():
    def __init__(self):
        self.handle = None
        self.info = None
        self.frame = None
        self.max_contacts = 5
        self.np_contacts = np.empty([self.max_contacts, 4], dtype=np.float32)
        self.np_contact_ids = np.empty([self.max_contacts], dtype=int)
        self.init()
    
    def init(self):
        # TODO: Better print messages
        self.load_sensel_lib()
        self.openSensel()
        if self.handle != None:
            self.getSensorInfo()
            self.initFrame()
            print('Sensel ready...')
        else:
            print('Sensel Open Error:', self.handle)
        
    def openSensel(self):
        handle = None
        (error, device_list) = self.getDeviceList()
        if device_list.num_devices != 0:
            (error, handle) = self.openDeviceByID(device_list.devices[0].idx)
        self.handle = handle
    
    def getSensorInfo(self):
        info = SenselSensorInfo(0,0,0,0,0)
        error = self._sensel_lib.senselGetSensorInfo(self.handle, byref(info))
        self.info = info
        print('Sensel got sensor info:', self.info)
        return error
    
    def initFrame(self):
        error = self.setFrameContent(FRAME_CONTENT_CONTACTS_MASK)
        (error, frame) = self.allocateFrameData()
        self.frame = frame
        print('Sensel allocated frame:', self.frame)
        error = self.startScanning()
        print('Sensel started scanning...')

    def scanFrames(self):
        error = self.readSensor()
        (error, num_frames) = self.getNumAvailableFrames()
        # print('Scan frames:', num_frames)
        for i in range(num_frames):
            error = self.getFrame()
            self.updateNpFrame()

    def updateNpFrame(self):
        # TODO: Add 'state' & other params
        if self.frame.n_contacts > 0:
            for n in range(self.frame.n_contacts):
                if n >= self.max_contacts:
                    self.np_contacts[:] = 0.0
                    self.np_contact_ids[:] = -1
                    break
                c = self.frame.contacts[n]
                self.np_contacts[n, 0] = c.x_pos
                self.np_contacts[n, 1] = self.info.height-c.y_pos
                self.np_contacts[n, 2] = c.area
                self.np_contacts[n, 3] = c.total_force
                self.np_contact_ids[n] = c.id
            # print('ids',self.np_contact_ids)
            # print('contacts',self.np_contacts)
        else:
            self.np_contacts[:,:] = 0.0
            self.np_contact_ids[:] = -1

    def printFrame(self):
        if self.frame.n_contacts > 0:
            for n in range(self.frame.n_contacts):
                c = self.frame.contacts[n]
                print("id: ", c.id,
                    'state:', c.state,
                    'x:', c.x_pos,
                    'y:', c.y_pos,
                    'total_force:', c.total_force,
                    'area:', c.area,
                    'orientation:', c.orientation)
                if c.state == CONTACT_START:
                    self.setLEDBrightness(c.id, 100)
                elif c.state == CONTACT_END:
                    self.setLEDBrightness(c.id, 0)

    def load_sensel_lib(self):
        self._sensel_lib = None
        self._sensel_lib_decompress = None
        platform_name = platform.system()
        if platform_name == "Windows":
            python_is_x64 = sys.maxsize > 2**32
            if python_is_x64:
                self._sensel_lib_decompress = windll.LoadLibrary("C:\\Program Files\\Sensel\\SenselLib\\x64\\LibSenselDecompress.dll")
                self._sensel_lib = windll.LoadLibrary("C:\\Program Files\\Sensel\\SenselLib\\x64\\LibSensel.dll")
            else:
                self._sensel_lib_decompress = windll.LoadLibrary("C:\\Program Files\\Sensel\\SenselLib\\x86\\LibSenselDecompress.dll")
                self._sensel_lib = windll.LoadLibrary("C:\\Program Files\\Sensel\\SenselLib\\x86\\LibSensel.dll")
        elif platform_name == "Darwin":
            self._sensel_lib = cdll.LoadLibrary("/usr/local/lib/libSensel.dylib")
        else:
            self._sensel_lib = cdll.LoadLibrary("/usr/lib/libsensel.so")

    def open(self):
        handle = c_void_p(0)
        error = self._sensel_lib.senselOpen(POINTER(handle))
        return (error, handle)

    def getDeviceList(self):
        device_list = SenselDeviceList(0)
        for i in range(SENSEL_MAX_DEVICES):
            device_list.devices[i] = SenselDeviceID(0)
        error = self._sensel_lib.senselGetDeviceList(byref(device_list))
        return (error, device_list)

    def openDeviceByID(self, idx):
        c_idx = c_ubyte(idx)
        handle = c_void_p(0)
        error = self._sensel_lib.senselOpenDeviceByID(byref(handle), c_idx)
        return (error, handle)

    def close(self):
        return self._sensel_lib.senselClose(self.handle)
        
    def softReset(self):
        return self._sensel_lib.senselSoftReset(self.handle)

    def allocateFrameData(self):
        frame_pointer = POINTER(SenselFrameData)()
        error = self._sensel_lib.senselAllocateFrameData(self.handle, byref(frame_pointer))
        return (error, frame_pointer.contents)

    def freeFrameData(self):
        error = self._sensel_lib.senselFreeFrameData(self.handle, byref(self.frame))
        return error

    def setScanDetail(self, detail):
        c_detail = c_int(detail)
        error = self._sensel_lib.senselSetScanDetail(self.handle, c_detail)
        return error

    def getScanDetail(self):
        detail = c_int(0)
        error = self._sensel_lib.senselGetScanDetail(self.handle, byref(detail))
        return (error, detail.value)

    def getSupportedFrameContent(self):
        content = c_ubyte(0)
        error = self._sensel_lib.senselGetSupportedFrameContent(self.handle, byref(content))
        return (error, content.value)

    def setFrameContent(self, content):
        c_content = c_ubyte(content)
        error = self._sensel_lib.senselSetFrameContent(self.handle, c_content)
        return error

    def getFrameContent(self):
        content = c_ubyte(0)
        error = self._sensel_lib.senselGetFrameContent(self.handle, byref(content))
        return (error, content.value)

    def startScanning(self):
        return self._sensel_lib.senselStartScanning(self.handle)

    def stopScanning(self):
        return self._sensel_lib.senselStopScanning(self.handle)

    def readSensor(self):
        return self._sensel_lib.senselReadSensor(self.handle)

    def getNumAvailableFrames(self):
        num_frames = c_int(0)
        error = self._sensel_lib.senselGetNumAvailableFrames(self.handle, byref(num_frames))
        return (error, num_frames.value)

    def getFrame(self):
        error = self._sensel_lib.senselGetFrame(self.handle, byref(self.frame))
        return error

    def setLEDBrightness(self, led_id, brightness):
        c_led_id = c_ubyte(led_id)
        c_brightness = c_ushort(brightness)
        error = self._sensel_lib.senselSetLEDBrightness(self.handle, c_led_id, c_brightness)
        return error

    def setContactsMask(self, mask):
        c_mask = c_ubyte(mask)
        error = self._sensel_lib.senselSetContactsMask(self.handle, c_mask)
        return error

    def getFrameContent(self):
        content = c_ubyte(0)
        error = self._sensel_lib.senselGetContactsMask(self.handle, byref(content))
        return (error, content.value)

    def readReg(self, reg, size):
        buf = (c_byte * size)()
        error = self._sensel_lib.senselReadReg(self.handle, c_ubyte(reg), c_ubyte(size), buf)
        return (error, buf)

    def writeReg(self, reg, size, data):
        buf = (c_ubyte * size)(*data)
        error = self._sensel_lib.senselWriteReg(self.handle, c_ubyte(reg), c_ubyte(size), buf)
        return error

    def readRegVS(self, reg, size):
        buf = (c_byte * size)()
        read_size = c_int(0)
        error = self._sensel_lib.senselReadRegVS(self.handle, c_ubyte(reg), c_ubyte(size), buf, byref(read_size))
        return (error, buf, read_size)

    def writeRegVS(self, reg, size, data):
        buf = (c_byte * size)(*data)
        write_size = c_int(0)
        error = self._sensel_lib.senselReadRegVS(self.handle, c_ubyte(reg), c_ubyte(size), buf, byref(write_size))
        return (error, write_size)
    
    def closeSensel(self):
        error = self.freeFrameData()
        error = self.stopScanning()
        error = self.close()

    def cleanup(self):
        self.closeSensel()
