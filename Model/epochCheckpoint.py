from keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
	def __init__(self, outputPath, every=1, startAt=0):
		# call the parent constructor
		super(Callback, self).__init__()
		self.outputPath = outputPath
		self.every = every
		self.intEpoch = startAt

	def on_epoch_end(self, epoch, logs={}):
		# check to see if the model should be serialized to disk
		if (self.intEpoch + 1) % self.every == 0:
			p = os.path.sep.join([self.outputPath,"best_model.h5"])
			# print("saving")
			self.model.save(p, overwrite=True)

		# increment the internal epoch counter
		self.intEpoch += 1
