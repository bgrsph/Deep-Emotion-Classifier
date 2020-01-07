from random import *
from domain.network import *

args = {}
args["test_mode"] = False
args["test_data_path"] = "/Users/bugrasipahioglu/PycharmProjects/EmotionClassification/dataset/test"
args["train_data_path"] = "/Users/bugrasipahioglu/PycharmProjects/EmotionClassification/dataset/train"
args["validation_data_path"] = "/Users/bugrasipahioglu/PycharmProjects/EmotionClassification/dataset/validation"
args["output_classes"] = ["sad", "happy", "disgust", "excited", "scared"]
args["model_name"] = "resnet34"
args["pretrained"] = "t"
args["progress"] = "t"
args["batch_size"] = 10
args["epochs"] = 15
args["learning_rate"] = 0.01
args["weight_decay"] = 0
args["momentum"] = 0
args["device"] = "cpu"
args["output_name"] = args["model_name"] + "-" + str(randint(0,100))

print("YOUR OUTPUT WILL BE: ", args["output_name"] + ".pth")
print("USE YOUR OUTPUT IN TEST")

print("Training")
Network(args=args).train()


