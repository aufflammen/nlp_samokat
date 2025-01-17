import gc
import time
import math
from collections.abc import Callable
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import torch

from utils import Ansi

def format_seconds_to_hms(seconds: int | float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


class CustomTrainer:

    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler,
        criterion: Callable | None = None, 
        metrics: Callable | None = None,
        main_metric: str | None = None,
        gradient_accumulation_steps: int = 1,
        max_steps: int | None = None,
        eval_steps: int | None = None,
        save_each_eval: bool = True,
        exp_path: str | None = None,
    ):
        self.micro_batch_size = train_loader.batch_size
        self.mini_batch_size = train_loader.batch_size * gradient_accumulation_steps
        self.global_steps_in_epoch = len(train_loader) * self.micro_batch_size / self.mini_batch_size

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.metrics = metrics
        self.main_metric = main_metric
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_steps = max_steps if max_steps is not None else math.ceil(len(self.train_loader) / self.mini_batch_size)
        self.eval_steps = eval_steps
        self.save_each_eval = save_each_eval
        self.exp_path = exp_path

        self.device = next(self.model.parameters()).device
        self.is_stop = False
        self.global_step = 0
        self.best = {
            'loss': None,
            'loss_step': None,
            'metric': None,
            'metric_step': None,
        }

        print(f"{'Micro batch:':<22} {Ansi.bold}{self.micro_batch_size}{Ansi.end}")
        print(f"{'Mini batch:':<22} {Ansi.bold}{self.mini_batch_size}{Ansi.end}\n")
        print(f"{'Total global steps:':<22} {Ansi.bold}{self.max_steps}{Ansi.end}")
        print(f"{'Global steps in epoch:':<22} {Ansi.bold}{self.global_steps_in_epoch:.1f}{Ansi.end}")
        print(f"{'Total epochs:':<22} {Ansi.bold}{self.max_steps / self.global_steps_in_epoch:.1f}{Ansi.end}")


    def _reset_init_metrics(self):
        self.loss_list = []
        if self.metrics is not None:
            self.metrics.reset()

    
    def _update_metrics(self, loss, outputs, labels):
        self.loss_list.append(loss)
        if self.metrics is not None:
            self.metrics.update(outputs, labels)

    
    def _compute_metrics(self):
        if self.metrics is None:
            return np.mean(self.loss_list), None
        else:
            return np.mean(self.loss_list), self.metrics.compute()

    
    def _model_save(self, filename):
        if self.exp_path is not None:
            checkpoint = self.model.state_dict()
            path = Path(self.exp_path)
            
            path.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path / f'{filename}.pth')

    
    def _save_best_model_by_loss_and_metric(self, loss, metrics):
        if self.best['loss'] is None or self.best['loss'] > loss:
            self.best['loss'] = loss
            self.best['loss_step'] = self.global_step
            if not self.save_each_eval:
                self._model_save('best_loss')

        if metrics is not None and self.main_metric is not None:
            if self.best['metric'] is None or self.best['metric'] < metrics[self.main_metric].item():
                self.best['metric'] = metrics[self.main_metric].item()
                self.best['metric_step'] = self.global_step
                # if not self.save_each_eval:
                    # self._model_save('best_metric')


    def evaluate(self, progress=True): 
        self._reset_init_metrics()
    
        self.model.eval()
        with torch.no_grad():
            if progress:
                pbar = tqdm(self.val_loader, desc='evaluation', leave=False)
            else:
                pbar = self.val_loader
    
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.criterion is None:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    labels = batch.pop('labels')
                    outputs = outputs[1]
                else:
                    labels = batch.pop('labels')
                    outputs = self.model(**batch)
                    loss = self.criterion(outputs, labels)

                self._update_metrics(loss.item(), outputs, labels)

        val_loss, val_metrics = self._compute_metrics()
        return val_loss, val_metrics

            
    def train(self, eval_progress=False):
        gc.collect()
        torch.cuda.empty_cache()

        self._reset_init_metrics()
        self.time_start = time.time()
        step = 0
        
        self.pbar = tqdm(range(self.max_steps))
        self.model.train()
        self.optimizer.zero_grad()
        while True:
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
    
                if self.criterion is None:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    labels = batch.pop('labels')
                    outputs = outputs[1]
                else:
                    labels = batch.pop('labels')
                    outputs = self.model(**batch)
                    loss = self.criterion(outputs, labels)
    
                (loss / self.gradient_accumulation_steps).backward()

                self._update_metrics(loss.item(), outputs, labels)
                step += 1

                # Gradient descent
                if step % self.gradient_accumulation_steps == 0:
                    step = 0
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
        
                    pbar_description = (
                        f"Step [{self.global_step}/{self.max_steps}], "
                        f"epoch: {self.global_step / self.global_steps_in_epoch:.2f} || "
                        f"lr: {self.scheduler.get_last_lr()[-1]:.1e}, loss: {np.mean(self.loss_list):.4f}, "
                        f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB"
                    )
                    self.pbar.set_description(pbar_description)
                    self.pbar.update(1)

                
                    if self.global_step % self.eval_steps == 0 or self.global_step == self.max_steps:
                        # Train metrics
                        train_loss, train_metrics = self._compute_metrics()

                        # Save model
                        if self.save_each_eval:
                            self._model_save(f'step_{self.global_step}')
        
                        # Valid
                        val_loss, val_metrics = self.evaluate(progress=eval_progress)
                        self._reset_init_metrics()
                        self.model.train()
        
                        # Save best model by loss and metric
                        self._save_best_model_by_loss_and_metric(val_loss, val_metrics)
        
                        # Log
                        self._log(train_loss, train_metrics, val_loss, val_metrics)
        
                        if self.global_step == self.max_steps:
                            self.is_stop=True
                            break

            if self.is_stop:
                break
        
        self._info_after_train()


    def _log(
        self, 
        train_loss, 
        train_metrics, 
        val_loss, 
        val_metrics
    ) -> str | None:

        padding = 8
        fmt_name = f"{{:>{padding}}}"
        fmt_value = f"{{:>{padding}.4f}}"
        fmt_value_loss = f"{Ansi.bold}{fmt_value}{Ansi.end}"
        fmt_value_lr = f"{{:>{padding}.1e}}"

        # Header
        if self.global_step == self.eval_steps:
            train_metric_names, _ = self._metriccollect2string(train_metrics, fmt_name, fmt_value)
            val_metric_names, _ = self._metriccollect2string(val_metrics, fmt_name, fmt_value)
            
            train_header = (
                f"{'step':>5} {fmt_name.format('lr')} | "
                f"{fmt_name.format('loss')} {train_metric_names}"
            )
            val_header = (
                f"{fmt_name.format('loss')} {val_metric_names}"
            )

            header_0 = (
                f"{'TRAIN'.center(len(train_header), '-')} | "
                f"{'VALID'.center(len(val_header), '-')}"
            )
            header_1 = (
                f"{Ansi.bold}{train_header} | {val_header}{Ansi.end}"
            )
            self.header_len = len(header_0)
            
            print(header_0)
            print(header_1)

        # Логгируемые параметры
        _, train_metric_values = self._metriccollect2string(train_metrics, fmt_name, fmt_value)
        _, val_metric_values = self._metriccollect2string(val_metrics, fmt_name, fmt_value)
        string = (
            f"{self.global_step:>5} {fmt_value_lr.format(self.scheduler.get_last_lr()[-1])} | "
            f"{fmt_value_loss.format(train_loss)} {train_metric_values} | "
            f"{fmt_value_loss.format(val_loss)} {val_metric_values}"
        )
        print(string)
        

    def _info_after_train(self):
        train_time = time.time() - self.time_start
        print('-' * self.header_len) # Линия, закрывающая таблицу
        print(f"{'Train time:':<14} {Ansi.bold}{format_seconds_to_hms(train_time)}{Ansi.end}")
        print(f"{'Global steps:':<14} {Ansi.bold}{self.global_step}{Ansi.end}")
        print(f"{'Best loss:':<14} {Ansi.bold}{self.best['loss']:.4f}{Ansi.end}, "
              f"step: {Ansi.bold}{self.best['loss_step']}{Ansi.end}")

        if self.main_metric is not None and self.metrics is not None:
            metric_name = f"Best {self.main_metric}:"
            print(f"{metric_name:<14} {Ansi.bold}{self.best['metric']:.4f}{Ansi.end}, "
                  f"step: {Ansi.bold}{self.best['metric_step']}{Ansi.end}")


    @staticmethod
    def _metriccollect2string(metrics_result: dict, fmtn, fmtv):
        if metrics_result is None:
            return '', ''

        names = " ".join(fmtn.format(key) for key in metrics_result.keys())
        values = " ".join(fmtv.format(value) for value in metrics_result.values())
        return names, values