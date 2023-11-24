import cv2

def draw_bboxes(image_path, bbox_txt_path):
    # Load the image
    image = cv2.imread(image_path)

    # Read COCO labels from txt file
    with open(bbox_txt_path, 'r') as f:
        annotations = []
        for line in f:
            bbox_data = line.strip().split(' ')
            x, y, w, h, class_id = [int(data) for data in bbox_data]
            annotations.append({
                'bbox': [x, y, w, h],
                'class_id': class_id
            })

    # Draw bounding boxes
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add class label
        class_label = 'Class: ' + str(ann['class_id'])
        cv2.putText(image, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)

if __name__ == '__main__':
    image_path = 'image.jpg'
    bbox_txt_path = 'coco_labels.txt'

    draw_bboxes(image_path, bbox_txt_path)