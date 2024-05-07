
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def load_model():
    labels = read_label_file('artifacts/labels.txt')
    interpreter = make_interpreter('artifacts/model.tflite')
    interpreter.allocate_tensors()
    return interpreter, labels