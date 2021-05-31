import cv2
import numpy as np

def draw_boxes(img, boxes, colors, required_idx = False, stride = 1, required_val= False):
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # org 
    org = (00, 185) 
    
    # fontScale 
    fontScale = 0.5

    # Line thickness of 2 px 
    thickness = 2

    for ii, box_float in enumerate(boxes):
        if len(box_float) >4:
            mid = box_float[4]
            color = colors[mid]
        else:
            color = colors[0]
        box = [int((float(xy)+0.5)*stride - 0.5) for xy in box_float[:4]]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        pos = [box[2], box[1]]
        if required_idx:
            text = '%d'%ii
            cv2.putText(image, text, pos[0], pos[1], font, fontScale,  color, thickness, cv2.LINE_AA, False)  

        if required_val and len(pos) > 2:
            text = '%d'%pos[2]
            cv2.putText(image, text, pos[0], pos[1], font, fontScale,  color, thickness, cv2.LINE_AA, False) 



def draw_circles(image, kpts_with_ids, colors, radius = 3, required_idx = False, stride = 1, required_val= False):

    
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # org 
    org = (00, 185) 
    
    # fontScale 
    fontScale = 0.5

    # Line thickness of 2 px 
    thickness = 2
    for ii, pos in enumerate(kpts_with_ids):
        if len(pos) >2:
            mid = pos[2]
            color = colors[mid]
        else:
            color = colors[0]
        cv2.circle(image, (int((pos[0]+0.5)* stride - 0.5), int((pos[1]+0.5)* stride - 0.5)), radius, color, 1, cv2.LINE_AA)
        if required_idx:
            text = '%d'%ii
            cv2.putText(image, text, (int(pos[0]* stride)+radius, int(pos[1]* stride)-radius), font, fontScale, 
                  color, thickness, cv2.LINE_AA, False)  

        if required_val and len(pos) > 2:
            text = '%d'%pos[2]
            cv2.putText(image, text, (int(pos[0]* stride)+radius, int(pos[1]* stride)-radius), font, fontScale, 
                  color, thickness, cv2.LINE_AA, False) 


def draw_polygon(image, order_corners, color, line_len = 15, stride =1, thickness = 2, is_skip_last = False, required_idx = False, idx_color = None, fontScale = 0.5):

    
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # org 
    org = (00, 185) 
    
    # fontScale 
    # fontScale = 0.5

    # Line thickness of 2 px 
    
    if idx_color is None:
        idx_color = color

    for ii in range(len(order_corners)):
        if ii ==0 and is_skip_last: continue
        if order_corners[ii-1] is None or order_corners[ii] is None: continue

        x1, y1 = (order_corners[ii-1][0]+0.5)* stride - 0.5, (order_corners[ii-1][1]+0.5)* stride - 0.5
        x2, y2 = (order_corners[ii][0]+0.5)* stride - 0.5, (order_corners[ii][1]+0.5)* stride - 0.5

        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)


        if required_idx:
            text = '%d'%ii
            cv2.putText(image, text, (int(x2), int(y2)), font, fontScale, 
                  idx_color, thickness, cv2.LINE_AA, False) 

def draw_vecs(image, kpts_with_ids, vecs, colors, line_len = 15, stride =1, inv_vec = False, style = None):

    if inv_vec:
        line_len *=-1
    # Line thickness of 2 px 
    thickness = 2
    for ii, pos in enumerate(kpts_with_ids):
        if len(pos) >2:
            mid = pos[2]
            color = colors[mid]
        else:
            color = colors[0]

        vec = vecs[ii]
        x1, y1 = (pos[0]+0.5)* stride - 0.5, (pos[1]+0.5)* stride - 0.5
        x2, y2 = x1 + vec[0] * line_len, y1 + vec[1] * line_len
        if style is None:
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)
        else:
            draw_dotted_line(image, (int(x1), int(y1)), (int(x2), int(y2)), color,thickness=1, style='dotted', gap=10)

def draw_dotted_line(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)

def resize_maps(heatmap, h,w):

    ch = heatmap.shape[2]
    heatmap_new = np.zeros([h,w,ch])
    for ii in range(ch):
        hm = heatmap[:,:,ii]
        heatmap_new[:,:,ii] = cv2.resize(hm, (w,h),interpolation=cv2.INTER_CUBIC)
        
    return heatmap_new


def draw_pose(img, rvecs, tvecs, cameraMatrix, distCoeffs, axis_size, is_inverse_z = False, rotate_idx = 0):

    axis_rotated = [[[1,0,0], [0,1,0]],
                    [[0,1,0], [-1,0,0]],
                    [[-1,0,0], [0,-1,0]],
                    [[0,-1,0], [1,0,0]],
                    ]
    axis = axis_rotated[rotate_idx] 


    if not is_inverse_z:
        axis += [[0,0,1]]
    else:    
        axis += [[0,0,-1]]
    axis =   np.float32(axis).reshape(-1,3) * axis_size
    corner = np.float32([[0,0,0]])
    # draw pose
    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cameraMatrix, distCoeffs)
    imgpts_corner, jac = cv2.projectPoints(corner, rvecs, tvecs, cameraMatrix, distCoeffs)
    corner = tuple(imgpts_corner[0].ravel())
    
    
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5)


    return img

def Draw3DCube(image, rvecs, tvecs, cameraMatrix, distCoeffs, axis_size, cube_color= None, line_width = 2):

    r = axis_size / 2
    objectPoints = [[-r, -r, 0],
                [r,-r,0],
                [r,r,0],
                [-r,r,0], 
                [-r, -r, -axis_size],
                [r,-r,-axis_size], 
                [r,r,-axis_size],
                [-r,r,-axis_size]]
    objectPoints = np.float32(objectPoints)
    imagePoints, jac = cv2.projectPoints(objectPoints, rvecs, tvecs, cameraMatrix, distCoeffs)

    if cube_color is None:
        cube_color = (0, 0, 255)

    # imagePoints = np.reshape(imagePoints, [-1, 1]).tolist()
    # imagePoints = imagePoints.ravel()

    # draw lines of different colours
    for i in range(4):
        
        cv2.line(image, tuple(imagePoints[i].ravel()), tuple(imagePoints[(i + 1) % 4].ravel()), cube_color, line_width)
        cv2.line(image, tuple(imagePoints[i + 4].ravel()), tuple(imagePoints[4 + (i + 1) % 4].ravel()), cube_color, line_width)
        cv2.line(image, tuple(imagePoints[i].ravel()), tuple(imagePoints[i + 4].ravel()), cube_color, line_width)

    return image