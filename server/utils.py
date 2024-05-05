
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def load_model():
    labels = read_label_file('models/coco_labels.txt')
    interpreter = make_interpreter('models/efficientdet_lite1_384_ptq_edgetpu.tflite')
    interpreter.allocate_tensors()
    return interpreter, labels