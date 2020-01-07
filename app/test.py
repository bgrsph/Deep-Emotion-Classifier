from domain.network import *

args = {}
WEIGHT_FOLDER_NAME = "vgg19-73.pth"

args["test_mode"] = True
args["test_data_path"] = "/Users/bugrasipahioglu/PycharmProjects/EmotionClassification/dataset/test"
args["pretrained_data_path"] = "/Users/bugrasipahioglu/PycharmProjects/EmotionClassification/trained_weights/" + WEIGHT_FOLDER_NAME
args["model_name"] = "vgg19"
args["output_classes"] = ["sad", "happy", "disgust", "excited", "scared"]
args["device"] = "cpu"
args["batch_size"] = 1

print("Testing...")
Network(args=args).test()
