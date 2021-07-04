







# test if train_metric (returned at each NAS it)
# is correlated with test_metric (accuracy from the test set)

def match_metrics(train_metric, test_metric, train_epochs):

		# sample and train 10 arch train_epochs epoch each
		# 		and store their train_metric in train_array
		# 		and compute their test_metric and store it in test_array

		train(best_epoch=0, epochs=10, epochs_child=train_epochs, show_time=0)



		# Plot the curve of each array + compute the variance