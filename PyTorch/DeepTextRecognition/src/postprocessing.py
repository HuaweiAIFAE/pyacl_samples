import numpy as np
#from constant import BLANK
BLANK = 0


class Process:
    
    def __init__(self, char_list, bboxes_sorted):
        self.charters = char_list
        self.bboxes = bboxes_sorted
        
    
    def text_boxes(self, preds, j):
        char_list = ' 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.charters = char_list
        pred_index = np.argmax(preds, axis=2)[0, :]    # output

        characters = []
        for i in range(preds.shape[1]):
            if pred_index[i] != BLANK and (not (i > 0 and pred_index[i - 1] == pred_index[i])):
                characters.append(self.charters[pred_index[i]])
            self.bboxes[j].text = ''.join(characters)
        #     text = ''.join(characters)
        # print(text)
           

    def get_concat_same(self):
        intersecting = []
        for i in range(len(self.bboxes) - 1):
            for j in range(i + 1, len(self.bboxes)):
                if (self.bboxes[i].is_horizontally_intersecting(self.bboxes[j])):
                    # print('i:{}, j:{} -> {}'.format(i, j, True), bboxes_sorted[i], bboxes_sorted[j])
                    intersecting.append([i, j])
        for idx, [i, j] in enumerate(intersecting):
            try:
                self.bboxes[i].merge_intersecting_box(self.bboxes[j])
            except :
                pass
            else:
                # print('Word corrected - ', bboxes_sorted[i], bboxes_sorted[j])
                self.bboxes.pop(j)
                for rest_idx in range(idx, len(intersecting)):
                    if intersecting[rest_idx][0] > j:
                        intersecting[rest_idx][0] -= 1
                    if intersecting[rest_idx][1] > j:
                        intersecting[rest_idx][1] -= 1
        return self.bboxes