from ultralytics import YOLO
import cv2
import numpy as np

#function to put multiline text on the frame
def put_text(frame, text, position, color=(255, 255, 255), font_scale=1, thickness=2):
    for i, line in enumerate(text.split('\n')):
        cv2.putText(frame, line, (position[0], position[1] + i*30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

class Annotator:
    """Annotator based on one from Ultralytics package."""

    def __init__(self, im, line_width=None, line_width_alert=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'

        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.0015), 2)  # line width
        self.lw_alert = line_width_alert or max(round(sum(im.shape) / 2 * 0.003), 2)

        
    def box_label(self, box, is_alert=False, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Add one xyxy box to image with label."""
        # if isinstance(box, torch.Tensor):
        try:
            box = box.tolist()
        except:
            pass

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw if not is_alert else self.lw_alert, lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        self.lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top', box_style=False):
        """Adds text to an image using PIL or cv2."""
        if anchor == 'bottom':  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        if self.pil:
            if box_style:
                w, h = self.font.getsize(text)
                self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            if '\n' in text:
                lines = text.split('\n')
                _, h = self.font.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += h
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(text, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = xy[1] - h >= 3
                p2 = xy[0] + w, xy[1] - h - 3 if outside else xy[1] + h + 3
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)  # filled
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            tf = max(self.lw - 1, 1)  # font thickness
            cv2.putText(self.im, text, xy, 0, self.lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

    def result(self):
        """Return annotated image as array."""
        return np.asarray(self.im)

model = YOLO('best_l.pt', task='detection')
img = 'datasets/urbanhack-train/images/0000000154building.jpg'

im0 = cv2.imread(img, cv2.IMREAD_COLOR)
im = im0.copy()
res = model.predict(im0)
det = res[0].boxes
# print(res)
ann = Annotator(im, line_width=1, font_size=0.1)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
for *bbox, conf, det_cls in det.data:
    ann.box_label(bbox, label=f'{res[0].names[int(det_cls)]}({int(conf*100)})', color=colors[int(det_cls)])
cv2.imwrite("res_l.png", ann.result())


