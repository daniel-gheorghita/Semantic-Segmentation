from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
import time
import itertools

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func()

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        parameters = itertools.ifilter(lambda p: p.requires_grad, model.parameters())
        #optim = self.optim(model.parameters(), **self.optim_args)
        optim = self.optim(parameters, **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        print 'START TRAIN.'
        ############################################################################
        # TODO:                                                                    #
        # Write your own personal training method for our solver. In Each epoch    #
        # iter_per_epoch shuffled training batches are processed. The loss for     #
        # each batch is stored in self.train_loss_history. Every log_nth iteration #
        # the loss is logged. After one epoch the training accuracy of the last    #
        # mini batch is logged and stored in self.train_acc_history.               #
        # We validate at the end of each epoch, log the result and store the       #
        # accuracy of the entire validation set in self.val_acc_history.           #
        #
        # Your logging should like something like:                                 #
        #   ...                                                                    #
        #   [Iteration 700/4800] TRAIN loss: 1.452                                 #
        #   [Iteration 800/4800] TRAIN loss: 1.409                                 #
        #   [Iteration 900/4800] TRAIN loss: 1.374                                 #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                                #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                                #
        #   ...                                                                    #
        ############################################################################
       
        #C = model.channels
        #W = model.width
        #H = model.height
    #    for epoch in range(num_epochs):
    #        for iteration in range(iter_per_epoch):
    #            loss = 0
    #            if (iteration % log_nth == 0):
    #                loss = 
    #                print '[Iteration {} / {}] TRAIN loss: {}'.format(iteration+1, iter_per_epoch, loss)
    #                self.train_loss_history.append(loss)
    #        train_acc = 0
    #        val_acc = 0
    #        self.train_acc_history.append(train_acc)
    #        self.val_acc_history.append(val_acc)
    #        print '[Epoch {} / {}] TRAIN acc/loss: {}/{}'.format(epoch+1, num_epochs, train_acc, loss)
    #        print '[Epoch {} / {}] VAL acc/loss: {}/{}'.format(epoch+1, num_epochs, val_acc, loss)
    #        print
        #print train_loader
        for epoch in range(num_epochs):
            #print 'Epoch start'
            tic = time.time()
            for iteration,data in enumerate(train_loader):
                #print 'Iteration start'
                inputs, labels = data

                # Wrap into Variable
                #inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                inputs, labels = Variable(inputs), Variable(labels)
                
                optim.zero_grad()


                # forward, backward, optimize
                outputs = model(inputs)
                
                (N,C,H,W) = outputs.data.size()
                
                outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous()
                outputs = outputs[labels.view(N, H, W, 1).repeat(1, 1, 1, C) >= 0].view(-1, C)

                labels_mask = labels >= 0
                labels = labels[labels_mask]
    
                loss = self.loss_func(outputs, labels)
                
                
                
                """
                outputs = outputs.view(N*H*W, C)
                labels = labels.view(N*H*W)

                labels_mask = labels >= 0
                
                labels_mask_index = Variable(torch.nonzero(labels_mask.data))
                
                
                labels_ok = labels_mask_index.data[:,0]
                labels_mask_index = Variable(labels_ok)
                
                outputsNew = outputs.index_select(0,labels_mask_index)
                labelsNew = labels.index_select(0,labels_mask_index)

                loss = self.loss_func(outputsNew, labelsNew)
                """
                loss.backward()
                optim.step()

                # Print statistics
                train_loss = loss.data[0]
                if (iteration % log_nth == 0):
                    print '[Iteration {} / {}] TRAIN loss: {}'.format(iteration+1, iter_per_epoch, train_loss)
                    self.train_loss_history.append(train_loss)
  
            """    
            # Training accuracy
            scores = []
            for batch in train_loader:
                #inputs, labels = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
                inputs, labels = Variable(batch[0]), Variable(batch[1])

                outputs = model(inputs)
                
                
                labels_mask = labels >= 0
                
                _, preds = torch.max(outputs, 1)

                scores.extend((preds == labels)[labels_mask].data.numpy())
            train_acc = np.mean(scores)
            self.train_acc_history.append(train_acc)
            print '[Epoch {} / {}] TRAIN acc/loss: {} / {}'.format(epoch+1, num_epochs, train_acc, train_loss)
            
            # Validation accuracy 
            scores = []
            for batch in val_loader:
                #inputs, labels = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
                inputs, labels = Variable(batch[0]), Variable(batch[1])
                (_,H,W) =  labels.data.size()
                outputs = model(inputs)
                
                labels_mask = labels >= 0
                
                _, preds = torch.max(outputs, 1)

                scores.extend((preds == labels)[labels_mask].data.numpy())
            val_acc = np.mean(scores)
            self.val_acc_history.append(val_acc)
            print '[Epoch {} / {}] VAL acc/loss: {} / {}'.format(epoch+1, num_epochs, val_acc, train_loss)
            """
            toc = time.time()
            print 'That took %fs' % (toc - tic)
            
            print
                
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        print 'FINISH.'
