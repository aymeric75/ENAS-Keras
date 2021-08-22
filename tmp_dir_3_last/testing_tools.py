import Controller as Con






# test if train_metric (returned at each NAS it)
# is correlated with test_metric (accuracy from the test set)

def match_metrics(train_metric, test_metric, train_epochs):

        # sample and train 10 arch train_epochs epoch each
        #       and store their train_metric in train_array
        #       and compute their test_metric and store it in test_array

        train(best_epoch=0, epochs=10, epochs_child=train_epochs, show_time=0)



        # Plot the curve of each array + compute the variance










    # if (show_time==1):
        
    #     time_train = time.time() - start
    #     time_epoch = time_train
    #     time_total = time_train*epochs           
        
    # epochs_tmp = range(len(val_loss))
    #     print("time one train: "+str(time_epoch)+"s or "+str(int(time_epoch/60))+" min")
    #     print("time for all epochs: "+str(time_total)+"s or "+str(int(time_total/60))+" min")
    #if (best_epoch==1 and show_time==0):
    #        plt.plot(epochs_tmp, val_loss, 'b')


