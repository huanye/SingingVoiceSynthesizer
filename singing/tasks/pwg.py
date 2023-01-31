import os

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DistributedSampler

import utils
from modules.stft_loss import MultiResolutionSTFTLoss
from parallel_wavegan.models import ParallelWaveGANDiscriminator, ParallelWaveGANGenerator
from parallel_wavegan.optimizers import RAdam
from tasks.base_task import BaseDataset, BaseTask
from utils import audio
from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDataset
from utils.pl_utils import data_loader


class EndlessDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle

        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = [i for _ in range(1000) for i in torch.randperm(
                len(self.dataset), generator=g).tolist()]
        else:
            indices = [i for _ in range(1000) for i in list(range(len(self.dataset)))]
        indices = indices[:len(indices) // self.num_replicas * self.num_replicas]
        indices = indices[self.rank::self.num_replicas]
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



class PwgDataset(BaseDataset):
    def __init__(self, data_dir, prefix, hparams, shuffle=False):
        super().__init__(data_dir, prefix, hparams, shuffle)
        self.data_dir = data_dir
        self.prefix = prefix
        self.shuffle = shuffle
        self.is_infer = prefix == 'test'
        self.batch_max_frames = 0 if self.is_infer else hparams['max_samples'] // hparams['hop_size']
        self.aux_context_window = hparams['generator_params']['aux_context_window']
        self.hop_size = hparams['hop_size']
        self.use_pitch_embed = hparams['generator_params']['use_pitch_embed']
        self.indexed_bs = None
        self.idx2key = np.load(f'{self.data_dir}/{self.prefix}_all_keys.npy')
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')

    @property
    def num_workers(self):
        return 2

    def _get_item(self, index):
        if self.indexed_bs is None:
            self.indexed_bs = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        item = self.indexed_bs[index]
        return item

    def __getitem__(self, index):
        item = self._get_item(index)
        key = self.idx2key[index]
        sample = {
            "id": index,
            "utt_id": key,
            "mel": item['mel'],
            "wav": item['wav'],
        }
        if self.use_pitch_embed:
            sample['pitch'] = item['pitch']
        return sample

    def collater(self, batch):
        if len(batch) == 0:
            return {}

        y_batch, c_batch, p_batch = [], [], []
        for idx in range(len(batch)):
            x, c = batch[idx]['wav'], batch[idx]['mel']
            if self.use_pitch_embed:
                p = batch[idx]['pitch']
            self._assert_ready_for_upsampling(x, c, self.hop_size, 0)
            if len(c) - 2 * self.aux_context_window > self.batch_max_frames:
                # randomly pickup with the batch_max_steps length of the part
                batch_max_frames = self.batch_max_frames if self.batch_max_frames != 0 else len(
                    c) - 2 * self.aux_context_window - 1
                batch_max_steps = batch_max_frames * self.hop_size
                interval_start = self.aux_context_window
                interval_end = len(c) - batch_max_frames - self.aux_context_window
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * self.hop_size
                y = x[start_step: start_step + batch_max_steps]
                c = c[start_frame - self.aux_context_window:
                      start_frame + self.aux_context_window + batch_max_frames]
                if self.use_pitch_embed:
                    p = p[start_frame - self.aux_context_window:
                          start_frame + self.aux_context_window + batch_max_frames]
                self._assert_ready_for_upsampling(y, c, self.hop_size, self.aux_context_window)
            else:
                # print(f"Removed short sample from batch (length={len(x)}).")
                continue
            y_batch += [torch.FloatTensor(y).reshape(-1, 1)]  # [(T, 1), (T, 1), ...]
            c_batch += [torch.FloatTensor(c)]  # [(T' C), (T' C), ...]
            if self.use_pitch_embed:
                p_batch += [torch.LongTensor(p)]  # [(T' C), (T' C), ...]

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = utils.collate_2d(y_batch, 0).transpose(2, 1)  # (B, 1, T)
        c_batch = utils.collate_2d(c_batch, 0).transpose(2, 1)  # (B, C, T')
        if self.use_pitch_embed:
            p_batch = utils.collate_1d(p_batch, 0)  # (B, T')
        else:
            p_batch = None

        # make input noise signal batch tensor
        z_batch = torch.randn(y_batch.size())  # (B, 1, T)
        return {
            'z': z_batch,
            'mels': c_batch,
            'wavs': y_batch,
            'pitches': p_batch
        }

    @staticmethod
    def _assert_ready_for_upsampling(x, c, hop_size, context_window):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == (len(c) - 2 * context_window) * hop_size


class PwgTask(BaseTask):
    def __init__(self):
        super(PwgTask, self).__init__()
        self.stft_loss = MultiResolutionSTFTLoss(use_mel_loss=hparams['use_mel_loss'])
        self.mse_loss_fn = torch.nn.MSELoss()
        self.l1_loss_fn = torch.nn.L1Loss()

    @data_loader
    def train_dataloader(self):
        train_dataset = PwgDataset(hparams['data_dir'], 'train', hparams, shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_sentences)

    @data_loader
    def val_dataloader(self):
        valid_dataset = PwgDataset(hparams['data_dir'], 'valid', hparams, shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_eval_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = PwgDataset(hparams['data_dir'], 'test', hparams, shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_eval_sentences)

    def build_dataloader(self, dataset, shuffle, max_sentences, endless=False):
        world_size = 1
        rank = 0
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        sampler_cls = DistributedSampler if not endless else EndlessDistributedSampler
        train_sampler = sampler_cls(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            collate_fn=dataset.collater,
            batch_size=max_sentences,
            num_workers=dataset.num_workers,
            sampler=train_sampler,
            pin_memory=True,
        )

    def configure_optimizers(self):
        set_hparams()
        self.build_model()
        optimizer_gen = RAdam(self.model_gen.parameters(),
                              **hparams["generator_optimizer_params"])
        optimizer_disc = RAdam(self.model_disc.parameters(),
                               **hparams["discriminator_optimizer_params"])
        self.scheduler = self.build_scheduler({'gen': optimizer_gen, 'disc': optimizer_disc})
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return {
            "gen": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer["gen"],
                **hparams["generator_scheduler_params"]),
            "disc": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer["disc"],
                **hparams["discriminator_scheduler_params"]),
        }

    def build_model(self):
        self.model_gen = ParallelWaveGANGenerator(ref_params=hparams.get('ref_params'),
                                                  **hparams["generator_params"])
        self.model_disc = ParallelWaveGANDiscriminator(**hparams["discriminator_params"])

    def _training_step(self, sample, batch_idx, optimizer_idx):
        z = sample['z']
        mels = sample['mels']
        y = sample['wavs']

        loss_output = {}
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            # calculate generator loss
            y_ = self.model_gen(z, mels, sample['pitches'], y)
            y, y_ = y.squeeze(1), y_.squeeze(1)
            sc_loss, mag_loss = self.stft_loss(y_, y)
            total_loss = sc_loss + mag_loss
            loss_output['sc'] = sc_loss
            loss_output['mag'] = mag_loss
            if self.global_step > hparams["discriminator_train_start_steps"]:
                # keep compatibility5
                total_loss *= hparams.get("lambda_aux_after_introduce_adv_loss", 1.0)
                p_ = self.model_disc(y_.unsqueeze(1))
                if not isinstance(p_, list):
                    # for standard discriminator
                    adv_loss = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_output['adv'] = adv_loss
                else:
                    # for multi-scale discriminator
                    adv_loss = 0.0
                    for i in range(len(p_)):
                        adv_loss += self.mse_loss_fn(
                            p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
                    adv_loss /= (i + 1)
                    loss_output['adv'] = adv_loss

                    # feature matching loss
                    if hparams["use_feat_match_loss"]:
                        # no need to track gradients
                        with torch.no_grad():
                            p = self.model_disc(y.unsqueeze(1))
                        fm_loss = 0.0
                        for i in range(len(p_)):
                            for j in range(len(p_[i]) - 1):
                                fm_loss += self.l1_loss_fn(p_[i][j], p[i][j].detach())
                        fm_loss /= (i + 1) * (j + 1)
                        loss_output["fm"] = fm_loss
                        adv_loss += hparams["lambda_feat_match"] * fm_loss
                total_loss += hparams["lambda_adv"] * adv_loss
        else:
            #######################
            #    Discriminator    #
            #######################
            if self.global_step > hparams["discriminator_train_start_steps"]:
                with torch.no_grad():
                    y_pred = self.model_gen(z, mels, sample['pitches'], y)
                # calculate discriminator loss
                p = self.model_disc(y)
                p_ = self.model_disc(y_pred)
                if not isinstance(p, list):
                    # for standard discriminator
                    real_loss = self.mse_loss_fn(p, p.new_ones(p.size()))
                    fake_loss = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                else:
                    # for multi-scale discriminator
                    real_loss = 0.0
                    fake_loss = 0.0
                    for i in range(len(p)):
                        real_loss += self.mse_loss_fn(p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                        fake_loss += self.mse_loss_fn(p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
                    real_loss /= (i + 1)
                    fake_loss /= (i + 1)
                loss_output["real"] = real_loss
                loss_output["fake"] = fake_loss
                total_loss = real_loss + fake_loss
            else:
                # skip disc training
                return None
        return total_loss, loss_output

    def on_after_backward(self):
        if self.opt_idx == 0:
            nn.utils.clip_grad_norm_(self.model_gen.parameters(), hparams['generator_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.model_disc.parameters(), hparams["discriminator_grad_norm"])

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()
        optimizer.zero_grad()
        if optimizer_idx == 0:
            self.scheduler['gen'].step(self.global_step // hparams['accumulate_grad_batches'])
        else:
            self.scheduler['disc'].step(
                (self.global_step - hparams["discriminator_train_start_steps"]) // hparams['accumulate_grad_batches'])

    def validation_step(self, sample, batch_idx):
        z = sample['z']
        mels = sample['mels']
        y = sample['wavs']
        pitch = sample['pitches']
        loss_output = {}

        #######################
        #      Generator      #
        #######################
        y_ = self.model_gen(z, mels, pitch, y)
        y, y_ = y.squeeze(1), y_.squeeze(1)
        sc_loss, mag_loss = self.stft_loss(y_, y)

        # p_ = self.model_disc(y_)
        # aux_loss = sc_loss + mag_loss
        # if self.global_step > hparams["discriminator_train_start_steps"]:
        #     # keep compatibility
        #     aux_loss *= hparams.get("lambda_aux_after_introduce_adv_loss", 1.0)
        # if not isinstance(p_, list):
        #     # for standard discriminator
        #     adv_loss = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
        #     gen_loss = aux_loss + hparams["lambda_adv"] * adv_loss
        # else:
        #     # for multi-scale discriminator
        #     adv_loss = 0.0
        #     for i in range(len(p_)):
        #         adv_loss += self.mse_loss_fn(
        #             p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
        #     adv_loss /= (i + 1)
        #     gen_loss = aux_loss + hparams["lambda_adv"] * adv_loss
        #
        #     # feature matching loss
        #     if hparams["use_feat_match_loss"]:
        #         p = self.model_disc(y.unsqueeze(1))
        #         fm_loss = 0.0
        #         for i in range(len(p_)):
        #             for j in range(len(p_[i]) - 1):
        #                 fm_loss += self.l1_loss_fn(p_[i][j], p[i][j])
        #         fm_loss /= (i + 1) * (j + 1)
        #         loss_output["fm"] += fm_loss.item()
        #         gen_loss += hparams["lambda_adv"] * hparams["lambda_feat_match"] * fm_loss

        # add to total eval loss
        loss_output["sc"] = sc_loss
        loss_output["mag"] = mag_loss
        # loss_output["gen"] = gen_loss
        # loss_output["adv"] = adv_loss
        # if self.global_step > hparams["discriminator_train_start_steps"]:
        #     #######################
        #     #    Discriminator    #
        #     #######################
        #     p = self.model_disc(y.unsqueeze(1))
        #     p_ = self.model_disc(y_.unsqueeze(1))
        #     if not isinstance(p_, list):
        #         # for standard discriminator
        #         real_loss = self.mse_loss_fn(p, p.new_ones(p.size()))
        #         fake_loss = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
        #         dis_loss = real_loss + fake_loss
        #     else:
        #         # for multi-scale discriminator
        #         real_loss = 0.0
        #         fake_loss = 0.0
        #         for i in range(len(p)):
        #             real_loss += self.mse_loss_fn(
        #                 p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
        #             fake_loss += self.mse_loss_fn(
        #                 p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
        #         real_loss /= (i + 1)
        #         fake_loss /= (i + 1)
        #         dis_loss = real_loss + fake_loss
        #     loss_output["real"] = real_loss
        #     loss_output["fake"] = fake_loss
        #     loss_output["disc"] = dis_loss
        # if dist.is_initialized():
        #     loss_output = utils.reduce_tensors(loss_output)
        loss_output = utils.tensors_to_scalars(loss_output)
        # for idx, (wav_pred, wav_gt) in enumerate(zip(y_, y)):
        #     wav_gt = wav_gt / wav_gt.abs().max()
        #     wav_pred = wav_pred / wav_pred.abs().max()
        #     self.logger.experiment.add_audio(f'valid/wav_{batch_idx}_{idx}_gt', wav_gt,
        #                                      self.global_step, sample_rate=22050)
        #     self.logger.experiment.add_audio(f'valid/wav_{batch_idx}_{idx}_pred', wav_pred,
        #                                      self.global_step, sample_rate=22050)
        return loss_output

    def _validation_end(self, outputs):
        all_losses_meter = {}
        for output in outputs:
            for k, v in output.items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = utils.AvgrageMeter()
                all_losses_meter[k].update(v, 1)
        loss_output = {k: round(v.avg, 4) for k, v in all_losses_meter.items()}
        loss_output['total_loss'] = loss_output['sc'] + loss_output['mag']
        return loss_output

    def test_step(self, sample, batch_idx):
        z = sample['z']
        mels = sample['mels']
        y = sample['wavs']
        pitch = sample['pitches']
        loss_output = {}
        y_ = self.model_gen(z, mels, pitch, y)
        gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}')
        os.makedirs(gen_dir, exist_ok=True)
        for idx, (wav_pred, wav_gt) in enumerate(zip(y_, y)):
            wav_gt = wav_gt / wav_gt.abs().max()
            wav_pred = wav_pred / wav_pred.abs().max()
            audio.save_wav(wav_gt.view(-1).cpu().float().numpy(), f'{gen_dir}/wav_{batch_idx}_{idx}_gt.wav', 22050)
            audio.save_wav(wav_pred.view(-1).cpu().float().numpy(), f'{gen_dir}/wav_{batch_idx}_{idx}_pred.wav', 22050)
        return loss_output

    def test_end(self, outputs):
        return {}

if __name__ == '__main__':
    PwgTask.start()
