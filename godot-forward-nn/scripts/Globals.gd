extends Node

# WIP
# MAKE SURE TO USE KERNEL SIZE = 2X2 FOR CONV2D

func _ready():
	
	print('xor inference test: ', example_1())
	print('conv2d forward test: ', example_2())
	
	pass
	
func example_1():
	
	# XOR inference. Weights & biases are loaded from model-1.json
	
	var batch = [[0, 0], [0, 1], [1, 0], [1, 1]]
	
	return Prediction.new('res://data//model-1.json').run(batch)[-1]

func example_2():
	
	# Runs an image tensor over the sequence
	# Conv2D -> MaxPooling2D -> Flatten -> Dense
	# and shows the result. Weights & biases are loaded from model-2.json
	
	var image_path = "res://images/randomimage1.png"
	var img = Image.new()
	img.load(image_path)
	img.lock()
	
	# Build image tensor with rows = height, cols = width, channels = 3
	var img_tensor_1 = Tensor.new(img.get_height(), img.get_width(), 3)
	var data_index = 0
	for i in range(img.get_height()):
		for j in range(img.get_width()):
			for k in range(3):
				img_tensor_1.set_item(i, j, k, img.get_data()[data_index])
				data_index += 1
					
	img.unlock()
	
	var batch = [img_tensor_1]
			
	return Prediction.new('res://data//model-2.json').run(batch)[-1]
	

	
