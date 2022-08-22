import cv2


def resizeframe(frame):
    scale = sizeScale(frame)
    #print(type(scale),scale)
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def sizeScale(frame):
    weight = int(frame.shape[1])
    height = int(frame.shape[0])
    #print(weight, height)
    if weight > height:
        if weight > 1920:
            #print(1920/weight)
            return float(1920/weight)
    elif weight < height:
        if height > 1080:
            #print(1080/height)
            return float(1080/height)
    else:
        return 1
