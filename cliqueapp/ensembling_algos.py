import numpy as np


def py_clique_ensemble(dets):

    def ensemble_boxes(boxes, label, type):
        box_arr = np.array(boxes)

        if type == 'max':
            score = np.max(box_arr[:, 1])
        if type == '1-x':
            score = 1 - np.prod(1 - box_arr[:, 1])

        box_arr[:, 1] = box_arr[:, 1] / np.sum(box_arr[:, 1])
        box_arr[:, 2:6] = np.multiply(box_arr[:, 2:6].T, box_arr[:, 1]).T
        final_bbox = np.sum(box_arr[:, 2:6], axis=0)

        return [label, score, final_bbox.item(0), final_bbox.item(1), final_bbox.item(2), final_bbox.item(3)]

    def iouvec(arr):
        x11, y11, x12, y12 = np.split(arr[:, [3, 2, 5, 4]], 4, axis=1)
        x21, y21, x22, y22 = np.split(arr[:, [3, 2, 5, 4]], 4, axis=1)

        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))

        interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
        boxAArea = (x12 - x11) * (y12 - y11)
        boxBArea = (x22 - x21) * (y22 - y21)

        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea + 1e-6)
        iou = np.maximum(iou, 0)
        iou = np.minimum(iou, 1)

        return iou

    def get_cliques(box_list):

        iou_arr = iouvec(np.array(box_list))
        edges = []

        # box [idx, float(Score), float(YMin), float(XMin), float(YMax), float(XMax)]

        for i in range(len(box_list)):
            for j in range(i):
                if iou_arr[i, j] > 0.5:
                    edges.append([i, j, iou_arr[i, j]])

        edges = sorted(edges, key=lambda edge: -edge[2])
        cliques = [v for v in range(len(box_list))]

        for edge in edges:
            clique0 = [i for i in range(len(box_list)) if cliques[i] == cliques[edge[0]]]
            clique1 = [i for i in range(len(box_list)) if cliques[i] == cliques[edge[1]]]
            colors0 = set(box_list[i][0] for i in clique0)
            colors1 = set(box_list[i][0] for i in clique1)

            if len(colors0.intersection(colors1)) == 0:
                for i in range(len(box_list)):
                    if cliques[i] == cliques[edge[0]]:
                        cliques[i] = cliques[edge[1]]

        cliques_dict = {}

        for clique in list(set(cliques)):
            cliques_dict[clique] = []

        for i in range(len(box_list)):
            cliques_dict[cliques[i]].append(box_list[i])

        return list(cliques_dict.values())

    boxes_dict = {'all': []}
    for i in range(dets.shape[0]):
        # box [idx, float(Score), float(YMin), float(XMin), float(YMax), float(XMax)]
        boxes_dict['all'].append([i, dets[i, 4], dets[i, 1], dets[i, 0], dets[i, 3], dets[i, 2]])

    result_boxes = []

    for key in boxes_dict.keys():
        cliques_result = get_cliques(boxes_dict[key])

        for clq in cliques_result:
            if len(clq) == 1:
                clq[0][0] = key
                result_boxes.append(clq[0])
            else:
                result_boxes.append(ensemble_boxes(clq, key, type='1-x'))

    result_boxes = sorted(result_boxes, key=lambda box: -box[1])

    return np.array(result_boxes)[:, [3, 2, 5, 4, 1]]


def py_greedy_nms(dets, iou_thr):
    """Pure python implementation of traditional greedy NMS.
    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        iou_thr (float): Drop the boxes that overlap with current
            maximum > thresh.
    Returns:
        numpy.array: Retained boxes.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_idx = scores.argsort()[::-1]

    keep = []
    while sorted_idx.size > 0:
        i = sorted_idx[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[sorted_idx[1:]])
        yy1 = np.maximum(y1[i], y1[sorted_idx[1:]])
        xx2 = np.minimum(x2[i], x2[sorted_idx[1:]])
        yy2 = np.minimum(y2[i], y2[sorted_idx[1:]])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (areas[i] + areas[sorted_idx[1:]] - inter)

        retained_idx = np.where(iou <= iou_thr)[0]
        sorted_idx = sorted_idx[retained_idx + 1]

    return dets[keep, :]


def py_soft_nms(dets, method='linear', iou_thr=0.5, sigma=0.5, score_thr=0.001):
    """Pure python implementation of soft NMS as described in the paper
    `Improving Object Detection With One Line of Code`_.
    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        method (str): Rescore method. Only can be `linear`, `gaussian`
            or 'greedy'.
        iou_thr (float): IOU threshold. Only work when method is `linear`
            or 'greedy'.
        sigma (float): Gaussian function parameter. Only work when method
            is `gaussian`.
        score_thr (float): Boxes that score less than the.
    Returns:
        numpy.array: Retained boxes.
    .. _`Improving Object Detection With One Line of Code`:
        https://arxiv.org/abs/1704.04503
    """
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 4], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1])

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        yy1 = np.maximum(dets[0, 1], dets[1:, 1])
        xx2 = np.minimum(dets[0, 2], dets[1:, 2])
        yy2 = np.minimum(dets[0, 3], dets[1:, 3])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 4] *= weight
        retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
        dets = dets[retained_idx + 1, :]

    return np.vstack(retained_box)


def algo_wrapper(boxes):
    boxes = np.delete(boxes, 4, 1)
    boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
    boxes = boxes.astype(float)

    res0 = py_clique_ensemble(boxes.copy())
    res1 = py_greedy_nms(boxes.copy(), 0.5)
    res2 = py_soft_nms(boxes.copy(), method='gaussian')

    print(res0.shape)
    print(res1.shape)
    print(res2.shape)

    colors_res = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    results = [res0, res1, res2]

    for i in range(len(results)):
        results[i] = results[i].astype(float)
        results[i][:, 2:4] = results[i][:, 2:4] - results[i][:, 0:2]
        tmp = results[i].copy()

        results[i] = []
        for j in range(tmp.shape[0]):
            results[i].append([int(tmp[j, 0]), int(tmp[j, 1]), int(tmp[j, 2]), int(tmp[j, 3]), colors_res[i], tmp[j, 4]])

    return results


if __name__ == '__main__':
    boxes_tmp = [[103, 23, 280, 365, (20, 208, 165), 0.15089694059786773], [104, 19, 263, 362, (169, 39, 43), 0.341790641961515], [118, 12, 258, 361, (70, 102, 220), 0.04638633757740773], [21, 27, 104, 339, (105, 21, 200), 0.49833093812676865], [24, 20, 98, 342, (196, 107, 74), 0.4886797516306577], [18, 16, 95, 356, (34, 27, 237), 0.11942282416867711]]
    results = algo_wrapper(boxes_tmp)
    print(results)
