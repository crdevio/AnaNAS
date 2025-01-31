# Autonomous Navigation And Neural Agent System

## How to use this project
- If you just use `python3 main.py`, your car will *learn* on the *default map*, with *no loaded pre-trained model* and *no file to save the trained model at the end*.
- Using `-l load_file` you precise the file where your model should *load the weights* (it need to exists otherwise it will not load).
- Using `-s save_file` you precise the file where your model should *save the trained weights* (it will save every 10 epochs by default, but you can change this, see the Constants section).
- Using `-t` you precise that you are in *test mode* (so no learning step).
- Using `-e` you precise the *first epsilon* (by default it's 1).

## The Constants
You need a lot of RAM really fast, to do some trade-of you can:
- Change the cone width and heights (it is the part of the map that your model see)
- The replay memory size (by default it's 10,000 but we recommend 100,000 to have efficient models).

If the training is too slow you can change increase the *\epsilon* decay (by default it's $10^{-3}$, we recommand $10^{-4}$ or even $10^{-5}$).

## The maps

You can make custom map and change it in the `STATIC_URL` (in `ai.py`).
