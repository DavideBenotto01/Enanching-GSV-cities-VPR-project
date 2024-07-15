import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
import pytorch_metric_learning
import utils
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from models import helper


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.
    """

    def __init__(self,
                 # ---- Backbone
                 backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=1,
                 layers_to_crop=[],

                 # ---- Aggregator
                 agg_arch='avg',  # GeM, AVG, MixVPR
                 agg_config={},

                 # ---- Train hyperparameters
                 lr=0.03,
                 optimizer='sgd',
                 scheduler = 'MultiStepLR',
                 weight_decay=1e-3,
                 momentum=0.9,
                 warmpup_steps=500,
                 milestones=[5, 10, 15],
                 lr_mult=0.3,

                 # ----- Loss
                 loss_name='MultiSimilarityLoss',
                 miner_name='MultiSimilarityMiner',
                 miner_margin=0.1,
                 faiss_gpu=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.save_hyperparameters()  # write hyperparams into a file

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []  # we will keep track of the % of trivial pairs/triplets at the loss level

        self.faiss_gpu = faiss_gpu

        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        # print("Descriptors shape:", x.shape)
        return x

    # configure the optimizer
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay,
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)
        if self.scheduler.lower() == "multisteplr":
            scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=self.lr_mult, milestones=self.milestones)
        elif self.scheduler.lower() == "cosineannealinglr":
            scheduler  = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
        elif self.scheduler.lower() == "cycliclr":
            scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.0002, max_lr=0.002, step_size_up=1, mode="triangular2")
        elif self.scheduler.lower() == "onecyclelr":
            scheduler = lr_scheduler.OneCycleLR(optimizer, epochs=15, steps_per_epoch=1, max_lr=0.002)
        else:
            raise ValueError(f'Scheduler {self.optimizer} has not been correctly implemented')
        self.optimizer = optimizer
        return [optimizer], [scheduler]

    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self, epoch, batch_idx,
                       optimizer, optimizer_idx, optimizer_closure,
                       on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)

            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)

        else:  # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                # somes losses do the online mining inside (they don't need a miner objet),
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class,
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                 len(self.batch_acc), prog_bar=True, logger=True)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch

        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape

        # reshape places and labels
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images)  # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels)  # Call the loss_function we defined above

        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}

    # This is called at the end of eatch training epoch
    def training_epoch_end(self, training_step_outputs):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        return descriptors.detach().cpu()

    def validation_epoch_end(self, val_step_outputs):
        dm = self.trainer.datamodule

        if len(dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            num_references = val_dataset.num_references
            num_queries = val_dataset.num_queries
            ground_truth = val_dataset.ground_truth

            r_list = feats[:num_references]
            q_list = feats[num_references:]

            recalls_dict, predictions = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 25],
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )

            del r_list, q_list, feats, num_references, ground_truth

            self.log(f'{val_set_name}/R1', recalls_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', recalls_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', recalls_dict[10], prog_bar=False, logger=True)
        print('\n\n')


if __name__ == '__main__':
    pl.utilities.seed.seed_everything(seed=1, workers=True)

    datamodule = GSVCitiesDataModule(
        batch_size=32,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=(320, 320),
        num_workers=8,
        show_data_stats=True,
        val_set_names=['sf_xs_val']
    )

    model = VPRModel(
        backbone_arch='resnet18',
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[4],
        agg_arch='gem',
        agg_config={'p': 3},
        # agg_config={'in_channels': 256, 'in_h': 20, 'in_w': 20, 'out_channels': 512, 'mix_depth': 4, 'mlp_ratio': 1, 'out_rows': 4},
        lr=0.0002, #adam
        # lr=1e-3,    #adamw
        # lr=0.3,   #asgd
        # lr=0.03,  #sgd
        scheduler = "MultiStepLR",
        optimizer='adam',
        weight_decay=0,
        momentum=0.9,
        warmpup_steps=600,
        milestones=[5, 10, 15, 25],
        lr_mult=0.3,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner',
        miner_margin=0.1,
        faiss_gpu=False
    )

    checkpoint_cb_sfxs_val = ModelCheckpoint(
        monitor='sf_xs_val/R1',
        filename=f'{model.encoder_arch}' +
                 'sf_xs_val_epoch({epoch:02d})_step({step:04d})_R1[{sf_xs_val/R1:.4f}]_R5[{sf_xs_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=-1,
        every_n_epochs=5,
        mode='max',
    )

    trainer = pl.Trainer(
        accelerator='gpu', devices=[0],
        default_root_dir=f'LOGS/{model.encoder_arch}',
        num_sanity_val_steps=0,
        precision=16,
        max_epochs=30,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb_sfxs_val],  # checkpoint_cb_sfxs_test, checkpoint_cb_tokyoxs],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
        # fast_dev_run=True # Remove this line to process the full dataset
    )

    trainer.fit(model=model, datamodule=datamodule)
