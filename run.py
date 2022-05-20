from module import TextSceneDetection
import argparse, logging
import matplotlib.pyplot as plt
import cv2

logging.basicConfig(filename=f'app.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')

parser = argparse.ArgumentParser("Unet Segmentation")
parser.add_argument('--i', type = str, required = True, default = False)
parser.add_argument('--o', type = str, required = False, default = 'out')


if __name__ == '__main__':
    args_parser = parser.parse_args()
    cls = TextSceneDetection("unet.onnx")
    logging.info("load model")
    image = cls(args_parser.i)
    cv2.imwrite(args_parser.o, image)
    plt.imshow(image)
    plt.show()
