import cv2
import torch
import numpy as np
from segment_anything import predictor
from segment_anything import SamPredictor, sam_model_registry


class ImageAnnotator:
    def __init__(self, image_path):
        #Init SAM
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_l"
        MODEL_PATH = "model/sam_vit_l_0b3195.pth"
        sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
        self.predictor = SamPredictor(sam)
        
        
        self.image_path = image_path
        self.points = []
        self.labels = []

        self.image = cv2.imread(self.image_path)
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        cv2.namedWindow("Image Annotator")
        cv2.imshow("Image Annotator", self.image)
        cv2.setMouseCallback("Image Annotator", self.on_mouse_click)

        while True:
            key = cv2.waitKey(0)

            if key == 13:  # Enter key
                pts_np = np.array(self.points)
                labels_np = np.array(self.labels)
                
                mask, scores, logits = self.predictor.predict(point_coords=pts_np, 
                                                               point_labels=labels_np,
                                                               multimask_output=False)
                mask_overlay = self.overlay_mask(self.image, mask)
                cv2.imshow("dst", mask_overlay)
                cv2.imwrite("result.jpg", mask_overlay)
                
            elif key == 27: # Esc key
                break

        cv2.destroyAllWindows()
        print("Annotation completed.")
        print("Points:", self.points)
        print("Key pressed:", self.key_pressed)

    def on_mouse_click(self, event, x, y, flags, param):
        #Update img
        img = self.image.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.labels.append(1)
            print(f"Postive Point added: ({x}, {y})")
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.points.append((x, y))
            self.labels.append(0)
            print(f"Negative Point added: ({x}, {y})")
        elif event == cv2.EVENT_MOUSEMOVE:
            #Draw aiming cross here
            img = cv2.line(img=img, pt1=(x-20,y), pt2=(x+20,y), color=(0,255,0),thickness=1)
            img = cv2.line(img=img, pt1=(x,y-20), pt2=(x,y+20), color=(0,255,0),thickness=1)
        
        for pt, label in zip(self.points, self.labels):
            color = (0,0,0)
            if label == 1:
                color = (0,255,0)
            else:
                color = (0,0,255)
            img = cv2.circle(img, center=pt, radius=2,color=color,thickness=2)
        cv2.imshow("Image Annotator", img)
    
    def overlay_mask(self, src, mask, color=(255, 0, 0), alpha=0.5):
        mask_t = np.transpose(mask, (1, 2, 0))
        mask_uint = mask_t.astype(np.uint8)*255
        print(np.mean(mask_uint))
        # 确保 mask 与原图大小相同
        mask_uint = cv2.resize(mask_uint, (src.shape[1], src.shape[0]))

        # 将颜色通道设置为指定颜色
        color_channel = np.zeros_like(src)
        color_channel[:, :, 0] = color[0]  # 蓝色通道
        color_channel[:, :, 1] = color[1]  # 绿色通道
        color_channel[:, :, 2] = color[2]  # 红色通道

        # 合并颜色通道和 mask
        color_mask = cv2.bitwise_and(color_channel, color_channel, mask=mask_uint)
        # 设置叠加权重
        beta = 1.0 - alpha  # mask权重
        cv2.imshow("mask", color_mask)
        # 使用addWeighted函数叠加原图和mask
        result = cv2.addWeighted(src, alpha, color_mask, beta, 0, 0)

        return result


if __name__ == "__main__":
    image_path = "example/2.jpg"
    
    annotator = ImageAnnotator(image_path)
