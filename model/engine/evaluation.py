import numpy as np
import torch
from tqdm import tqdm
from util.logger import *
from util.visdom_plots import VisdomLogger

def do_evaluation(
        cfg,
        model,
        dataloader,
        device,
        visdom
):
    model.eval()
    logger = get_logger('eval')
    logger.info("Start evaluating")

    loss_records = []
    with torch.no_grad():
        for iteration, (images, steering_commands, _) in tqdm(enumerate(dataloader)):
            images = images.to(device)
            steering_commands = steering_commands.to(device)
            predictions, loss = model(images, steering_commands)
            loss_records.append(loss.item())

            if iteration % cfg.LOG.PERIOD == 0:
                visdom.update({'eval_loss': [loss.item()]})
                logger.info("EVAL_LOSS: \t{}".format(loss))

    visdom.plot('eval_loss')

    logger.info('LOSS EVALUATION: {}'.format(np.mean(loss_records)))

