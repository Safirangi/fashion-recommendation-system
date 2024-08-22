import requests
import base64
import numpy as np
from PIL import Image
import io
import cv2

API_URL = "https://api-inference.huggingface.co/models/mattmdjaga/segformer_b2_clothes"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def get_clothes(img_path,choice):
    output = query(img_path)
    # Load original image
    original_image = cv2.imread(img_path)

    for part in output:
        if choice=="top":
            if part['label']!='Skirt' and part['label']!='Pants':
                
                continue
        elif choice=='bottom':
            if part['label']!='Upper-clothes':
                continue
        # print(part)
        base64_encoded_mask = part['mask']
        decoded_bytes = base64.b64decode(base64_encoded_mask)
        mask_image = np.array(Image.open(io.BytesIO(decoded_bytes)))

        # Resize mask to match original image size
        mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]))

        # Threshold the mask to create a binary mask
        _, binary_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)
        
        # Invert binary mask
        binary_mask = cv2.bitwise_not(binary_mask)

        # Create white background image
        white_background = np.ones_like(original_image) * 255

        # Copy original image where mask is white
        result_image = np.where(binary_mask[..., None], white_background, original_image)

        # cv2.imshow("masked_img",result_image)
        # cv2.waitKey(3000)
        # Save the result image
        # cv2.imwrite(f"masked_area.jpg", result_image)

# print(query("/home/saiganesh.s/ML/realtime/captured_image.jpg"))
if __name__=='__main__':
    get_clothes('/home/saiganesh.s/ML/realtime/captured_image.jpg',"bottom")
    