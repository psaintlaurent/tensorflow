# 2.3.1

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_image(i, predictions_array, true_label, img):
	true_label, img = true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img, cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel(
		"{} {}% ({})".format(class_names[predicted_label],
							 100 * np.max(predictions_array),
							 class_names[true_label],
							 color=color
							 )
	)


def plot_value_array(i, predictions_array, true_label):
	true_label = true_label[i]
	plt.grid(False)
	plt.xticks(range(10))
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data() 


#print(tf.__version__)

class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

print("Training Images Shape: ", training_images.shape)
print("Training Labels Length ", (len(training_labels)))
print("Training Labels Data ", (training_labels))
print("Test Images ", (test_images.shape))
print("Test Label Length ", (len(test_labels)))


#plt.figure()
#plt.imshow(training_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(training_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[training_labels[i]])
#plt.show()


model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28,28)),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(10)
])


model.compile(
	optimizer='adam', 
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
	metrics=['accuracy']
)


model.fit(training_images, training_labels, epochs=20)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


probability_model = tf.keras.Sequential([
	model,
	tf.keras.layers.Softmax()
])

#predictions = probability_model.predict(test_images)

#for i in range(10):
#	print(predictions[i])
#	print(np.argmax(predictions[i]), class_names[np.argmax(predictions[i])])

#for i in range(10):
#	plt.figure(figsize=(6,3))
#	plt.subplot(1,2,1)
#	plot_image(i, predictions[i], test_labels, test_images)
#	plt.subplot(1,2,2)
#	plot_value_array(i, predictions[i], test_labels)
#	plt.show()


#predict the correct label for an image
img = test_images[1]
print(img.shape)

img = (np.expand_dims(img, 0))
print(img.shape)

predictions_single = probability_model.predict(img)
print("Single Prediction", predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

np.argmax(predictions_single[0])
