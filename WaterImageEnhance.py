 from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import cv2
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
from keras.models import model_from_json
from utils.data_utils import read_and_resize, preprocess, deprocess
from DataReader import DataReader

# ================= GUI SETUP =================
main = tkinter.Tk()
main.title("Underwater Image Enhancement with Multi-Scale Residual Attention Network")
main.geometry("1200x950")

multi_patch_model = None
residual_model = None
trainData = None
testData = None
itr = None
dataset_path = None

# ================= PSNR =================
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse))

# ================= SSIM =================
def imageSSIM(original, enhanced):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    return ssim(original_gray, enhanced_gray,
                data_range=enhanced_gray.max() - enhanced_gray.min())

# ================= UPLOAD DATASET =================
def uploadDataset():
    global dataset_path, trainData, testData, itr

    text.delete('1.0', END)

    dataset_path = filedialog.askdirectory(initialdir=".")
    if dataset_path == "":
        return

    dr = DataReader()

    try:
        trainImages = dr.readImages('Dataset/reference-890')
        testImages = dr.readImages('Dataset/raw-890')

        trainData, testData, itr = dr.generateTrainTestImages(trainImages, testImages)

        text.insert(END, "Training & Testing Images Loaded Successfully\n\n")
        text.insert(END, "Total Training Images : " + str(len(trainImages)) + "\n")
        text.insert(END, "Total Testing Images  : " + str(len(testImages)) + "\n")

        pathlabel.config(text="UIEB Dataset Loaded")

    except Exception as e:
        text.insert(END, "Error Loading Dataset:\n")
        text.insert(END, str(e))

# ================= LOAD MULTI PATCH MODEL =================
def loadMultiPatch():
    global multi_patch_model

    text.delete('1.0', END)

    try:
        with open('models/gen_p/model_15320_.json', "r") as json_file:
            loaded_model_json = json_file.read()

        multi_patch_model = model_from_json(loaded_model_json)
        multi_patch_model.load_weights('models/gen_p/model_15320_.h5')

        pathlabel.config(text="Multi-Patch Model Loaded")
        text.insert(END, "Pretrained Multi-Patch Model Loaded Successfully\n")

    except Exception as e:
        text.insert(END, "Error Loading Multi-Patch Model:\n")
        text.insert(END, str(e))

# ================= RESIDUAL MODEL =================
def buildResidualModel():
    inputs = Input(shape=(256,256,3))
    x = Conv2D(64,3,padding='same',activation='relu')(inputs)
    x = Conv2D(64,3,padding='same',activation='relu')(x)
    x = Conv2D(3,3,padding='same')(x)
    outputs = Add()([inputs,x])
    model = Model(inputs, outputs)
    return model

def loadResidualModel():
    global residual_model

    text.delete('1.0', END)

    residual_model = buildResidualModel()
    residual_model.compile(optimizer='adam', loss='mse')

    pathlabel.config(text="Residual Enhancement Model Loaded")
    text.insert(END, "Residual Enhancement Model Loaded Successfully\n")

# ================= ENHANCE IMAGE =================
def uploadImage():
    global multi_patch_model, residual_model

    filename = askopenfilename(initialdir=".")
    if filename == "":
        return

    text.delete('1.0', END)
    pathlabel.config(text=filename)

    orig = cv2.imread(filename)
    orig = cv2.resize(orig,(256,256))

    inp_img = read_and_resize(filename,(256,256))
    im = preprocess(inp_img)
    im = np.expand_dims(im,axis=0)

    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    images = [orig_rgb]
    titles = ["Original Image"]

    # ===== Multi-Patch Prediction =====
    if multi_patch_model is not None:
        gen1 = multi_patch_model.predict(im)
        enhance1 = deprocess(gen1)[0]
        enhance1 = cv2.resize(enhance1,(256,256))
        enhance1_rgb = cv2.cvtColor(enhance1, cv2.COLOR_BGR2RGB)

        psnr1 = PSNR(orig, enhance1)
        ssim1 = imageSSIM(orig, enhance1)

        text.insert(END,"Multi-Patch Model Results:\n")
        text.insert(END,"PSNR : "+str(round(psnr1,2))+" dB\n")
        text.insert(END,"SSIM : "+str(round(ssim1,4))+"\n\n")

        images.append(enhance1_rgb)
        titles.append("Multi-Patch Output")

    # ===== Residual Prediction =====
    if residual_model is not None:
        gen2 = residual_model.predict(im)
        enhance2 = deprocess(gen2)[0]
        enhance2 = cv2.resize(enhance2,(256,256))
        enhance2_rgb = cv2.cvtColor(enhance2, cv2.COLOR_BGR2RGB)

        psnr2 = PSNR(orig, enhance2)
        ssim2 = imageSSIM(orig, enhance2)

        text.insert(END,"Residual Model Results:\n")
        text.insert(END,"PSNR : "+str(round(psnr2,2))+" dB\n")
        text.insert(END,"SSIM : "+str(round(ssim2,4))+"\n")

        images.append(enhance2_rgb)
        titles.append("Residual Output")

    # ===== SHOW IMAGES =====
    fig, axes = plt.subplots(1, len(images), figsize=(5*len(images),5))

    if len(images) == 1:
        axes = [axes]

    for i in range(len(images)):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

# ================= CLOSE =================
def close():
    main.destroy()

# ================= GUI DESIGN =================
font = ('times', 20, 'bold')
title = Label(main, text='Underwater Image Enhancement with Multi-Scale Residual Attention Network')
title.config(bg='brown', fg='white', font=font, height=3, width=80)
title.place(x=5,y=5)

font1 = ('times', 14, 'bold')

btn0 = Button(main, text="Upload UIEB Dataset", command=uploadDataset)
btn0.place(x=50,y=80)
btn0.config(font=font1)

btn1 = Button(main, text="Generate & Load Multi-Patch Model", command=loadMultiPatch)
btn1.place(x=50,y=130)
btn1.config(font=font1)

btn2 = Button(main, text="Load Residual Enhancement Model", command=loadResidualModel)
btn2.place(x=50,y=180)
btn2.config(font=font1)

btn3 = Button(main, text="Upload & Enhance Image", command=uploadImage)
btn3.place(x=50,y=230)
btn3.config(font=font1)

btn4 = Button(main, text="Exit", command=close)
btn4.place(x=50,y=280)
btn4.config(font=font1)

pathlabel = Label(main, bg='brown', fg='white', font=font1)
pathlabel.place(x=420,y=150)

text = Text(main,height=15,width=100)
text.place(x=50,y=350)

main.config(bg='brown')
main.mainloop()