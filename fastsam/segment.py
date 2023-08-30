import os, sys, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .utils import image_to_np_ndarray


class FastSAMSegment:

    def __init__(self, image, results, device='cuda'):
        if isinstance(image, str) or isinstance(image, Image.Image):
            image = image_to_np_ndarray(image)
        self.device = device
        self.results = results
        self.img = image

    def _segment_image(self, image, bbox):
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        segmented_image_array = np.zeros_like(image_array)
        x1, y1, x2, y2 = bbox
        segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
        segmented_image = Image.fromarray(segmented_image_array)
        black_image = Image.new('RGB', image.size, (255, 255, 255))
        # transparency_mask = np.zeros_like((), dtype=np.uint8)
        transparency_mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
        transparency_mask[y1:y2, x1:x2] = 255
        transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
        black_image.paste(segmented_image, mask=transparency_mask_image)
        return black_image


    def plot_to_result(self,
             annotations,
             bboxes=None,
             points=None,
             point_label=None,
             mask_random_color=True,
             better_quality=True,
             retina=False,
             withContours=True) -> np.ndarray:
        if isinstance(annotations[0], dict):
            annotations = [annotation['segmentation'] for annotation in annotations]
        image = self.img
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h = image.shape[0]
        original_w = image.shape[1]
        if sys.platform == "darwin":
            plt.switch_backend("TkAgg")
        plt.figure(figsize=(original_w / 100, original_h / 100))
        # Add subplot with no margin.
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.imshow(image)
        if better_quality:
            if isinstance(annotations[0], torch.Tensor):
                annotations = np.array(annotations.cpu())
            for i, mask in enumerate(annotations):
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
        if self.device == 'cpu':
            annotations = np.array(annotations)
            self.fast_show_mask(
                annotations,
                plt.gca(),
                random_color=mask_random_color,
                bboxes=bboxes,
                points=points,
                pointlabel=point_label,
                retinamask=retina,
                target_height=original_h,
                target_width=original_w,
            )
        else:
            if isinstance(annotations[0], np.ndarray):
                annotations = torch.from_numpy(annotations)
            self.fast_show_mask_gpu(
                annotations,
                plt.gca(),
                random_color=mask_random_color,
                bboxes=bboxes,
                points=points,
                pointlabel=point_label,
                retinamask=retina,
                target_height=original_h,
                target_width=original_w,
            )
        if isinstance(annotations, torch.Tensor):
            annotations = annotations.cpu().numpy()
        if withContours:
            contour_all = []
            temp = np.zeros((original_h, original_w, 1))
            for i, mask in enumerate(annotations):
                if type(mask) == dict:
                    mask = mask['segmentation']
                annotation = mask.astype(np.uint8)
                if not retina:
                    annotation = cv2.resize(
                        annotation,
                        (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                contours, hierarchy = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour_all.append(contour)
            cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
            color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
            contour_mask = temp / 255 * color.reshape(1, 1, -1)
            plt.imshow(contour_mask)

        plt.axis('off')
        fig = plt.gcf()
        plt.draw()

        try:
            buf = fig.canvas.tostring_rgb()
        except AttributeError:
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
        cols, rows = fig.canvas.get_width_height()
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols, 3)
        result = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        plt.close()
        return result


    # Remark for refactoring: IMO a function should do one thing only, storing the image and plotting should be seperated and do not necessarily need to be class functions but standalone utility functions that the user can chain in his scripts to have more fine-grained control.
    def plot(self,
             annotations,
             output_path,
             bboxes=None,
             points=None,
             point_label=None,
             mask_random_color=True,
             better_quality=True,
             retina=False,
             withContours=True):
        if len(annotations) == 0:
            return None
        result = self.plot_to_result(
            annotations,
            bboxes,
            points,
            point_label,
            mask_random_color,
            better_quality,
            retina,
            withContours,
        )

        path = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(path):
            os.makedirs(path)
        result = result[:, :, ::-1]
        cv2.imwrite(output_path, result)


    def box_prompt(self, bbox=None, bboxes=None):
        if self.results == None:
            return []
        assert bbox or bboxes
        if bboxes is None:
            bboxes = [bbox]
        max_iou_index = []
        for bbox in bboxes:
            assert (bbox[2] != 0 and bbox[3] != 0)
            masks = self.results[0].masks.data
            target_height = self.img.shape[0]
            target_width = self.img.shape[1]
            h = masks.shape[1]
            w = masks.shape[2]
            if h != target_height or w != target_width:
                bbox = [
                    int(bbox[0] * w / target_width),
                    int(bbox[1] * h / target_height),
                    int(bbox[2] * w / target_width),
                    int(bbox[3] * h / target_height), ]
            bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
            bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
            bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
            bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

            # IoUs = torch.zeros(len(masks), dtype=torch.float32)
            bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

            masks_area = torch.sum(masks[:, bbox[1]:bbox[3], bbox[0]:bbox[2]], dim=(1, 2))
            orig_masks_area = torch.sum(masks, dim=(1, 2))

            union = bbox_area + orig_masks_area - masks_area
            IoUs = masks_area / union
            max_iou_index.append(int(torch.argmax(IoUs)))
        max_iou_index = list(set(max_iou_index))
        return np.array(masks[max_iou_index].cpu().numpy())


    def fast_show_mask_gpu(
        self,
        annotation,
        ax,
        random_color=False,
        bboxes=None,
        points=None,
        pointlabel=None,
        retinamask=True,
        target_height=960,
        target_width=960,
    ):
        msak_sum = annotation.shape[0]
        height = annotation.shape[1]
        weight = annotation.shape[2]
        areas = torch.sum(annotation, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=False)
        annotation = annotation[sorted_indices]
        # Find the index of the first non-zero value at each position.
        index = (annotation != 0).to(torch.long).argmax(dim=0)
        #if random_color:
        #    color = torch.rand((msak_sum, 1, 1, 3)).to(annotation.device)
        #else:
        #    color = torch.ones((msak_sum, 1, 1, 3)).to(annotation.device) * torch.tensor([
        #        30 / 255, 144 / 255, 255 / 255]).to(annotation.device)
        color = torch.ones((msak_sum, 1, 1, 3)).to(annotation.device) * torch.tensor([
            30 / 255, 144 / 255, 255 / 255]).to(annotation.device)
        transparency = torch.ones((msak_sum, 1, 1, 1)).to(annotation.device) * 1.0
        visual = torch.cat([color, transparency], dim=-1)
        mask_image = torch.unsqueeze(annotation, -1) * visual
        # Select data according to the index. The index indicates which batch's data to choose at each position, converting the mask_image into a single batch form.
        show = torch.zeros((height, weight, 4)).to(annotation.device)
        try:
            h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight), indexing='ij')
        except:
            h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
        indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
        # Use vectorized indexing to update the values of 'show'.
        show[h_indices, w_indices, :] = mask_image[indices]
        show_cpu = show.cpu().numpy()
        #if bboxes is not None:
        #    for bbox in bboxes:
        #        x1, y1, x2, y2 = bbox
        #        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
        if not retinamask:
            show_cpu = cv2.resize(show_cpu, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        show_cpu = 1 - show_cpu
        ax.imshow(show_cpu)
