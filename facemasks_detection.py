import cv2
import numpy as np

YOLO_WEIGHTS = 'weights/tiny-yolo-facemasks_best.weights'
YOLO_CFG = 'cfg/tiny-yolo-facemasks.cfg'

NET = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)

classes = {0: 'no mask', 1: 'mask'}
colors = {0: (75, 74, 224), 1: (229, 160, 21)}
conf_threshold = 0.6
nms_threshold = 0.4
scale = 0.00392

cap = cv2.VideoCapture(0)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def get_yolo_result(image: np.ndarray):
    image_height = image.shape[0]
    image_width = image.shape[1]

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    boxes = []
    confidences = []
    class_ids = []
    NET.setInput(blob)
    outs = NET.forward(get_output_layers(NET))
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                w = int(detection[2] * image_width)
                h = int(detection[3] * image_height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    return class_ids, boxes, confidences

def draw_prediction(image: np.ndarray, class_id: int, confidence: float, box: list):
    box = list(map(round, box))
    x, y, w, h = box
    color = colors[class_id]
    confidence = str(round(confidence, 2))
    label = classes[class_id] + ' ' + confidence
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
    cv2.rectangle(image, (x, y - 25), (x + w, y), color, -1)
    cv2.putText(image, label, (x + 5, y - 5), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def main():
    while True:
        _, frame = cap.read()
        
        class_ids, boxes, confidences = get_yolo_result(frame)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            frame = draw_prediction(frame, class_id, confidence, box)
            
        key = cv2.waitKey(1)
        cv2.imshow('camera', frame)
        if key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
