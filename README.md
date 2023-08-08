# ML Dog Doodler

## About
A web app designed for you to get a custom doodle of your pup. 🐶
* Backend: A Pytorch-based image classifier that is trained and tested on the Stanford Dogs dataset. The model is a convolutional neural network based on the VGG16 architecture (with a few modifications like extra dropout layers) with 75% accuracy on 5 breeds. There's also a CNN with a custom-designed architecture that ended up being about 10% less accurate. The VGG16 pth file is fed to a Flask server, wherein ten popular dog breeds are classifiable.
* Frontend: A React app that asks you for an image of your pup, and returns to you the most accurate hand-drawn doodle of them:

<img width="1247" alt="dog doodler img" src="https://github.com/yashadoddabele/dog-doodler/assets/110857917/2ae97458-4258-405b-aef4-949f31b4f039">

## Tech Used
* React
* Flask
* PyTorch
