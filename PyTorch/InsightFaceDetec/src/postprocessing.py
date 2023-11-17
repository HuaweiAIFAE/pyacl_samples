import numpy as np


class PostProcess():

    def __init__(self,
                       model_output_dims,
                       feat_stride_fpn,
                       nms_thresh,
                       fms,
                       fmb,
                       num_anchors,
                       batched,
                       use_kps):
        self.model_output_dims = model_output_dims
        self.feat_stride_fpn = feat_stride_fpn
        self.nms_thresh = nms_thresh
        self.fms = fms
        self.fmb = fmb
        self.num_anchors = num_anchors
        self.batched = batched
        self.use_kps = use_kps
        self.center_cache = {}
    
        
    
    def detect(self, output_data, det_scale, input_height, input_width, thresh, max_num=0, metric='default'):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        for idx, stride in enumerate(self.feat_stride_fpn):
                # 0, 2, 4 -> bboxes
                # 1, 3, 5 -> scores
                # If model support batch dim, take first output
                if self.batched:
                    scores = output_data[idx][0]
                    bbox_preds = output_data[idx+3][0]
                    bbox_preds = bbox_preds * stride
                

                    if self.use_kps: # to do (it might be some bugs)
                        kps_preds = self.__get_model_output_by_index(output_data, idx + self.fmc * 2)[0] * stride
                # If model doesn't support batching take output as is
                else:
                    scores = output_data[idx]
                    bbox_preds = output_data[idx+3]
                    bbox_preds = bbox_preds * stride                    

                    if self.use_kps: # to do (it might be some bugs)
                        kps_preds = self.__get_model_output_by_index(output_data, idx + self.fmc * 2) * stride
                
                height = input_height // stride
                width = input_width // stride
                K = height * width
                key = (height, width, stride)
                
                if key in self.center_cache:
                    anchor_centers = self.center_cache[key]
                else:
                    #solution-1, c style:
                    #anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                    #for i in range(height):
                    #    anchor_centers[i, :, 1] = i
                    #for i in range(width):
                    #    anchor_centers[:, i, 0] = i

                    #solution-2:
                    #ax = np.arange(width, dtype=np.float32)
                    #ay = np.arange(height, dtype=np.float32)
                    #xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                    #anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                    #solution-3:
                    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                    #print(anchor_centers.shape)
                    anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                    
                    if self.num_anchors>1:
                        anchor_centers = np.stack([anchor_centers]*self.num_anchors, axis=1).reshape( (-1,2) )
                    if len(self.center_cache)<100:
                        self.center_cache[key] = anchor_centers

                pos_inds = np.where(scores>=thresh)[0]
                bboxes = self.__distance2bbox(anchor_centers, bbox_preds)
                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]
                scores_list.append(pos_scores)
                bboxes_list.append(pos_bboxes)

                if self.use_kps:
                    kpss = distance2kps(anchor_centers, kps_preds)
                    #kpss = kps_preds
                    kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                    pos_kpss = kpss[pos_inds]
                    kpss_list.append(pos_kpss)
        # get detections
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale

        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self. __nms(pre_det)
        det = pre_det[keep, :]

        if self.use_kps:
            kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering

            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]

            if kpss is not None:
                kpss = kpss[bindex, :]

        return det, kpss
    
    
    def __nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
    
    
    def __distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
            
        return np.stack([x1, y1, x2, y2], axis=-1)