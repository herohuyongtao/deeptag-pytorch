import cv2
import numpy as np
from util.visualization import *


def visualize_rt(image, rvecs, tvecs, cameraMatrix, distCoeffs, tag_real_size_in_meter, tag_id_decimal, tid_text_pos = None, score=None, fontScale = 0.5, thickness = 2, is_draw_cube = True, rotate_idx = 0, text_color = None ):
    if text_color is None:
        text_color = (0,196,196)
    
    if score is not None:
        # draw tag_id, score
        tid_text = 'ID, score: %d, %.2f'%(tag_id_decimal, score)
    elif tag_id_decimal >=0:
        # draw tag_id
        tid_text = 'ID=%d'%(tag_id_decimal)
    else:
        tid_text = None
    # tid_text_pos = kpts_with_ids_in_image[0]
    if tid_text_pos is None:
        objectPoints = np.float32([[0,0,0]])
        imagePoints, jac = cv2.projectPoints(objectPoints, rvecs, tvecs, cameraMatrix, distCoeffs)
        tid_text_pos = imagePoints[0].ravel().tolist()

  

    # draw xyz-axis
    image = draw_pose(image,rvecs, tvecs, cameraMatrix, distCoeffs, tag_real_size_in_meter/2, rotate_idx = rotate_idx)
    if is_draw_cube:
        # draw cube
        image =Draw3DCube(image,rvecs, tvecs, cameraMatrix, distCoeffs, tag_real_size_in_meter)

    if tid_text is not None:
        cv2.putText(image, tid_text ,(int(tid_text_pos[0]), int(tid_text_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale, text_color,thickness,cv2.LINE_AA)

    return image

def vis_tag_rois(image_in, tag_res, stride_hm =1):
    image = image_in.copy()
    # draw tag boundary
    # colors = [(256, 256, 64), (64, 256, 64), (256, 64, 64)] + [(10, 10, 128)] *2
    colors = [(256, 256, 64)]*10
    idx_color = (128, 128, 32)
    for tag_detect_info in tag_res:
        color = colors[tag_detect_info['tag_id']]
        ordered_corners =tag_detect_info['ordered_corners']
        draw_polygon(image, ordered_corners, color, line_len = 15, stride =stride_hm, required_idx= True, idx_color= idx_color)

    return image

def visualize_correct_rate(image, valid_count, count):
    h, w = image.shape[:2]
    text_pos = w -300, 30
    rate_text = 'Detection: %d/%d'%(valid_count, count)
    # tid_text_pos = kpts_with_ids_in_image[0]
    cv2.putText(image, rate_text ,(int(text_pos[0]), int(text_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(128,255,128),2,cv2.LINE_AA)

    return image


def vis_tag_anchor_rois(image_in, tag_res, stride_hm =1):
    image = image_in.copy()
    # draw tag boundary
    # colors = [(256, 256, 64), (64, 256, 64), (256, 64, 64)] + [(10, 10, 128)] *2
    colors = [(256, 256, 64)]*10
    idx_color = (128, 128, 32)
    anchor_color = (0, 0, 192)
    idx_anchor_color = (0, 0, 128)

    for tag_detect_info in tag_res:
        color = colors[tag_detect_info['tag_id']]
        ordered_corners =tag_detect_info['ordered_corners']
        draw_polygon(image, ordered_corners, color, line_len = 15, stride =stride_hm, required_idx= True, idx_color= idx_color, fontScale = 1)
        ordered_corners =tag_detect_info['corner_anchors']
        draw_polygon(image, ordered_corners, anchor_color, line_len = 15, stride =stride_hm, required_idx= True, idx_color= idx_anchor_color, fontScale = 1)

    return image

def vis_center_and_corners(image_in, image_res, center_to_corner_links = [], stride = 1, stride_hm =1):
    image = image_in.copy()
    # h,w = image.shape[:2]

    # draw corners
    corners_with_ids = image_res['corners_with_ids']
    colors = [(128, 255, 255)]*5
    draw_circles(image, corners_with_ids, colors, stride= stride_hm, radius = 5)

    colors = [(0, 0, 255)]*5
    corner_directions = image_res['corner_directions']
    draw_vecs(image, corners_with_ids, corner_directions, colors, stride= stride_hm, inv_vec = True)

    

    # draw centers
    centers_with_ids= image_res['centers_with_ids']
    colors = [(256, 256, 64), (64, 256, 64), (256, 64, 64)] + [(10, 10, 128)] *2
    draw_circles(image, centers_with_ids, colors, stride= stride, radius = 7, required_val = False)


    # draw bbox

    colors = [(256, 256, 64), (64, 256, 64), (256, 64, 64)] + [(10, 10, 128)] *2
    draw_boxes(image, image_res['boxes'], colors)
    anchors_set = image_res['anchors_in_boxes']

    count = 0
    colors = [(256, 64, 64)]
    for anchors in anchors_set:  
        count +=1
        # if count %10 !=1:continue
        draw_circles(image, anchors, colors, radius=7, required_idx= True)


    # draw link from center to corner
    colors = [(256, 256, 64), (64, 256, 64), (256, 64, 64)] + [(10, 10, 128)] *2
    for ii, center_with_id in enumerate(centers_with_ids):
        if len(center_to_corner_links)<=ii: break       
        

        center_pos = center_with_id[:2]        
        links = [[corner[jj] - center_pos[jj] for jj in range(2)] for corner in center_to_corner_links[ii]]

        pts_with_ids = [corner[:2] + [center_with_id[2]] for corner in center_to_corner_links[ii]]

        draw_vecs(image, pts_with_ids, links, colors, stride= stride_hm, inv_vec = True, line_len = 1, style = 'dotted')


    return image


def visual_keypoints(image, decoded_tag, tag_id_decimal = -1, upsample_scale = 1):


        image_rect = cv2.warpPerspective(image, decoded_tag['H_crop'], decoded_tag['image_size_crop'])
        basic_width = image_rect.shape[0]//50

        # draw image and template
        image_rect_out = cv2.resize(image_rect, None, fx =upsample_scale, fy=upsample_scale) 

        fontScale = 0.7
        thickness= 2
        tid_text_pos = [20* upsample_scale,20* upsample_scale]

        display_ratio = image_rect.shape[0]/256
        # print(display_ratio)
        if display_ratio>1:
            
            fontScale = fontScale * display_ratio
            thickness = int(thickness*display_ratio)
            tid_text_pos = [20*display_ratio *upsample_scale,20*display_ratio* upsample_scale]

        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 196, 196), (128, 128, 128)]
        draw_circles(image_rect_out, decoded_tag['keypoints_with_ids'], colors, required_idx= False ,radius=int(basic_width*0.4*upsample_scale), stride=upsample_scale)

        if tag_id_decimal >=0 :
            # draw tag_id
            tid_text = 'ID=%d'%(tag_id_decimal)            
            cv2.putText(image_rect_out, tid_text ,(int(tid_text_pos[0]), int(tid_text_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale,(0,196,196), thickness,cv2.LINE_AA)
      


        return image_rect_out