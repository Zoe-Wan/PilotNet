import os
import torch
from model.engine.evaluation import do_evaluation
from datetime import datetime
from util.logger import *

from util.visdom_plots import VisdomLogger


def do_train(
        cfg,
        model,
        dataloader_train,
        dataloader_evaluation,
        optimizer,
        device
):
    # set mode to training for model (matters for Dropout, BatchNorm, etc.)


    # get the trainer logger and visdom
    visdom = VisdomLogger(cfg.LOG.PLOT.DISPLAY_PORT)
    visdom.register_keys(['train_loss', 'eval_loss'])
    logger = get_logger('train')
    logger.info("Start training")

    output_dir = os.path.join(cfg.LOG.PATH, 'run_{}'.format(datetime.now().strftime("%Y-%m-%d_%H.%M.%S")))
    os.makedirs(output_dir)

    # start the training loop
    for epoch in range(cfg.SOLVER.EPOCHS):
        model.train()
        for iteration, (images, steering_commands, _) in enumerate(dataloader_train):
            images = images.to(device)
            steering_commands = steering_commands.to(device)

            predictions, loss = model(images, steering_commands)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % cfg.LOG.PERIOD == 0:
                visdom.update({'train_loss': [loss.item()]})
                logger.info("TRAIN_LOSS: \t{}".format(loss))

            step = epoch * len(dataloader_train) + iteration
            if step % cfg.LOG.WEIGHTS_SAVE_PERIOD == 0 and iteration:
                torch.save(model.state_dict(),
                           os.path.join(output_dir, 'weights_{}.pth'.format(str(step))))
        visdom.plot('train_loss')
        do_evaluation(cfg, model, dataloader_evaluation, device, visdom)

    torch.save(model.state_dict(), os.path.join(output_dir, 'weights_final.pth'))
