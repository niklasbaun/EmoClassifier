import csv
import time



"""
Main predict function

"""
def predict(csv):
    ##TODO: Implement the prediction logic here
    print("Doing Something")
    #loading spiral
    animation = "|/-\\"
    idx = 0
    for idx in range(100):
        print(animation[idx % len(animation)], end="\r")
        idx += 1
        time.sleep(0.1)

predict(csv)

