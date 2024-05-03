from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def load_model():
    # labels = read_label_file('models/coco_labels.txt')
    interpreter = make_interpreter('models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
    interpreter.allocate_tensors()
    return interpreter