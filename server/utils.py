
import yaml
import numpy as np
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import cv2
import collections


Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])
BBox = collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])


def read_yaml():
    try:
        CONFIG_PATH = 'configs/model_config.yaml'
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print("Failed to read yaml file. {}".format(CONFIG_PATH))
        raise e
    return config

config = read_yaml()

def convert_bbox(bbox):

    x, y, w, h = tuple(bbox)

    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    return xmin, ymin, xmax, ymax


def set_inputs(image):
    """
    Pre process input image

    Args:
        image (PIL): PIL Image

    Returns:
        np.ndarray: scaled input tensor array
    """
    numpy_image = np.array(image).astype(np.float32)

    input_tensor = np.expand_dims(numpy_image, axis = 0)

    scaled_tensor = np.divide(input_tensor, 255)
    
    if config['input']['shape']['preprocessing']['normalization']:
        normalization = config['input']['shape']['preprocessing']['normalization']
        return np.clip(scaled_tensor * normalization[1],
                       normalization[0],
                       normalization[1]).astype(np.int8)
    return scaled_tensor

def detection(interpreter):

    output_tensor = postprocess(interpreter)

    pp_params = config['output']['postprocessing']

    dims = np.nonzero(output_tensor[:,4:] > pp_params["top_k"])
    actual_dims = dims[0], dims[1] + 4, dims[2]
    classes = dims[1]
    scores = output_tensor[actual_dims]
    boxes = output_tensor[0, :4, dims[-1]]

    idx = cv2.dnn.NMSBoxes(boxes, scores,
                           pp_params["confidence_threshold"],
                           pp_params["iou_threshold"])
    boxes = (boxes * config['input']['shape'][1]).astype(np.int32)
    def make(i):
        xmin, ymin, xmax, ymax = convert_bbox(boxes[i])
        return Object(id=int(classes[i]), score=float(scores[i]),
                      bbox = BBox(xmin=xmin,
                                  ymin=ymin,
                                  xmax=xmax,
                                  ymax=ymax))

    return [make(i) for i in range(len(idx))]


def postprocess(interpreter):

    quantization_params = config['output']['postprocessing']["quantization"]
    if len(config['output']['tensor_id']) == 1:
        output_tensor = interpreter.get_tensor(config['output']['tensor_id'][0])
    else:
        raise ValueError("Detection not implemented for this model. {}".format(config['output']['tensor_id']))
    return (output_tensor.astype(np.float32) - quantization_params["zero_point"]) * quantization_params["scales"]


def init_model(config):

    labels = read_label_file(config['output']['label_file_path'])
    interpreter = make_interpreter(config['model']['path'])
    interpreter.allocate_tensors()

    return interpreter, labels
