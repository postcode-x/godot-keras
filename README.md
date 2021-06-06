# Godot & Keras
**Using Keras weights & biases inside Godot projects**

This project is currently divided in two parts: 

1) A python project to train a model and solve XOR. The network architecture consists of a sequence of Dense layers with different # of nodes each, which is not necessary in the case of XOR but it serves as an example. After the training is completed the weights & biases are saved inside a JSON file, which we can open using the provided Godot project to perform forward passes (predictions) using the same Dense layer architecture from the keras model. 

2) A python project to test Conv2D, MaxPooling2D and Flatten layers by pushing an image Tensor through the model. Weights and biases are only initialized (no training is performed), so after running the python script they are ready to be loaded inside the same Godot project and perform a forward pass, in this case through the sequence Conv2D -> MaxPooling2D -> Flatten -> Dense. 

**This project is a W.I.P.** and for now it only supports Dense, Flatten, Conv2D (partially) and MaxPooling2D (partially) layers. Kernels & PoolSizes only work when they are of dimension 2 x 2.

Keep in mind that **I'm not planning to add Backpropagation**. I strongly think that game engines should only perform inference tasks with previously trained models because straining the CPU or GPU by doing Backpropagation during gameplay (which is not a visual activity at all) is just not worth it unless the problem is extremely simple, in which case the solution might not even need an artificial neural network at all.
