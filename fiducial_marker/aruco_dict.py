import cv2
from cv2 import aruco
import numpy as np

# def get_arucotag_info(dictInfo, tag_id):


def ArucoBits(bytesList, markerSize):

    arucoBits = np.zeros(shape = (markerSize, markerSize), dtype = bool)
    base2List = np.array( [128, 64, 32, 16, 8, 4, 2, 1], dtype = np.uint8)
    currentByteIdx = 0
    currentByte = bytesList[currentByteIdx]
    currentBit = 0
    for row in range(markerSize):
        for col in range(markerSize):
            if(currentByte >= base2List[currentBit]):
                arucoBits[row, col] = 1
                currentByte -= base2List[currentBit]
            currentBit = currentBit + 1
            if(currentBit == 8):
                currentByteIdx = currentByteIdx + 1
                currentByte = bytesList[currentByteIdx]
                if(8 * (currentByteIdx + 1) > arucoBits.size):
                    currentBit = 8 * (currentByteIdx + 1) - arucoBits.size
                else:
                    currentBit = 0;
    return arucoBits


def get_all_aruco_dict_flags():
    # Name, Flag
    dict_flags = \
    {
        "DICT_4X4_1000": aruco.DICT_4X4_1000,
        "DICT_5X5_1000": aruco.DICT_5X5_1000,
        "DICT_6X6_1000": aruco.DICT_6X6_1000,
        "DICT_7X7_1000": aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11,
    }

    # Name, tag_size     
    dict_sizes = {}
    for k in dict_flags.keys():
        arucoDict_flag = dict_flags[k]
        arucoDict = aruco.Dictionary_get(arucoDict_flag)
        grid_size = arucoDict.markerSize
        dict_size = arucoDict.bytesList.shape[0]
        dict_sizes[k] = {'grid_size': grid_size, 'dict_size': dict_size}

    return dict_flags, dict_sizes

def get_aruco_info(arucoDict_flag, tag_id):
    arucoDict = aruco.Dictionary_get(arucoDict_flag)
    grid_size = arucoDict.markerSize
    arucoBits = ArucoBits(arucoDict.bytesList[tag_id].ravel(), grid_size)
    return arucoBits

if __name__ == '__main__':
    dict_flags, dict_sizes = get_all_aruco_dict_flags()
    for k in dict_flags.keys():
        arucoBits = get_aruco_info(dict_flags[k], 0)
        print(arucoBits.astype(np.uint8).ravel().tolist())
    