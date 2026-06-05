import torch
import torch.nn.functional as F
import pytorch_lightning as pl
# from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import argparse
import random
import json

from scipy import stats

import domino.models.domino_models
from domino.data.domino_data import GenomeDataset

import matplotlib.pyplot as plt

def process_pos_neg(df, pred):
    df_duplicated = pd.DataFrame(df.values.repeat(2, axis=0), columns=df.columns)
    df_duplicated['direction'] = ['+'] * len(df_duplicated)
    df_duplicated['direction'] = df_duplicated.groupby(df_duplicated.index // 2)['direction'].transform(lambda x: ['+', '-'] * (len(x) // 2))
    df_duplicated['pred'] = pred
    for col in ['start', 'end', 'boundary']:
        df_duplicated[col] = df_duplicated[col].astype(int)
    for col in ['ins_score_1', 'ins_score_2', 'demeaned_ins_score_1', 'demeaned_ins_score_2']:
        df_duplicated[col] = df_duplicated[col].astype(float)
    return df_duplicated

if __name__ == '__main__':
    print("cuda is available: ", torch.cuda.is_available())
    parser = argparse.ArgumentParser(description="Model Hyperparameters")

    parser.add_argument(
        "--model_type", type=str, default="DominoSingleOutput",
    )
    parser.add_argument("--out_folder", help="the output folder")

    # Data

    parser.add_argument("--genome_path")
    parser.add_argument("--ins1_path")
    parser.add_argument("--ins2_path")
    parser.add_argument("--bound_path")
    parser.add_argument("--blacklist_path", type=str, help='path to file with blacklisted regions')
    parser.add_argument("--delta", help="delta parameter for boundary calling", default=2)
    parser.add_argument("--ins_window_size", help="insulation window size (in kb)")
    parser.add_argument("--demean_context_bin_width", help="number of context bins in (one direction) used for demean operation", type=int, default=10)
    parser.add_argument("--DNA_context_bin_width", help="number of context bins in (one direction) used as input", type=int, default=10)
    parser.add_argument("--boundary_length", help="length of each example (in bp)", type=int, default=200)
    parser.add_argument("--batch_size", help="the batch size", type=int, default=32)
    
    parser.add_argument("--train_coords_dict", help="chromosomes for training set", type=json.loads)
    parser.add_argument("--val_coords_dict", help="chromosomes for val set", type=json.loads)
    parser.add_argument("--test_coords_dict", help="chromosomes for test set", type=json.loads)

    parser.add_argument("--shuffle_seq_TRAIN", default=False)
    parser.add_argument("--shuffle_seq_VAL", default=False)

    # Model
    parser.add_argument("--model_random_seed", default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="plateau")   # cycle, plateau

    parser.add_argument(
        "--conv_layers",
        help="number of the large convolutional layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--conv_repeat",
        help="number of the convolutional conv blocks",
        type=int,
        default=1,
    )

    parser.add_argument("--kernel_number", help='number of initial filters', type=int, default=1024)
    parser.add_argument("--kernel_length", help='size of initial conv filter', type=int, default=17)
    parser.add_argument("--filter_number", help='number of intermediate filters', type=int, default=256)
    parser.add_argument("--filter_size", help='size of intermediate conv filters', type=int, default=5)

    parser.add_argument("--h_layers", help='number of hidden layers', type=int, default=2)
    parser.add_argument("--hidden_size", help='hidden dimension', type=int, default=256)
    
    parser.add_argument("--num_workers", help='number of workers', type=int, default=6)
    
    parser.add_argument("--activation", help='activation function', default='gelu')


    args = parser.parse_args()

    model_type = args.model_type
    out_folder = args.out_folder

    # Data

    genome_path = args.genome_path
    ins1_path = args.ins1_path
    ins2_path = args.ins2_path
    bound_path = args.bound_path
    blacklist_path = args.blacklist_path
    delta = args.delta
    ins_window_size = args.ins_window_size

    demean_context_bin_width = int(args.demean_context_bin_width)
    DNA_context_bin_width = int(args.DNA_context_bin_width)
    boundary_length = int(args.boundary_length)
    batch_size = args.batch_size

    train_coords_dict = args.train_coords_dict
    val_coords_dict = args.val_coords_dict
    test_coords_dict = args.test_coords_dict

    print(type(train_coords_dict))
    print(type(val_coords_dict))
    print(type(test_coords_dict))

    shuffle_seq_TRAIN = True if args.shuffle_seq_TRAIN == 'True' else False
    shuffle_seq_VAL = True if args.shuffle_seq_VAL == 'True' else False

    # Model

    model_random_seed = int(args.model_random_seed)
    learning_rate = float(args.learning_rate)
    scheduler = args.scheduler
    activation = args.activation

    conv_layers = int(args.conv_layers)
    conv_repeat = int(args.conv_repeat)

    kernel_number = int(args.kernel_number)
    kernel_length = int(args.kernel_length)
    filter_number = int(args.filter_number)
    filter_size = int(args.filter_size)

    h_layers = int(args.h_layers)
    hidden_size = int(args.hidden_size)

    num_workers = int(args.num_workers)

    output_path = (
        out_folder + 
        "/"
        + model_type
        + "_"
        + str(demean_context_bin_width) 
        + "_"
        + str(DNA_context_bin_width)
        + "_"
        + str(delta)
        + "_"
        + str(ins_window_size)
        + "_"
        + str(boundary_length)
        + "_"
        + str(batch_size)
        + "_"
        + str(shuffle_seq_TRAIN)
        + "_"
        + str(shuffle_seq_VAL)
        + "_"
        + str(model_random_seed)
        + "_"
        + str(learning_rate)
        + "_"
        + str(scheduler)
        + "_"
        + str(conv_layers)
        + "_"
        + str(conv_repeat)
        + "_"
        + str(kernel_number)
        + "_"
        + str(kernel_length)
        + "_"
        + str(filter_number)
        + "_"
        + str(filter_size)
        + "_"
        + str(h_layers)
        + "_"
        + str(hidden_size)
        + "_"
        + str(activation)
    )

    checkpoint_path = output_path + "/checkpoint"
    log_path = out_folder + "/log"

    # to ensure reproducibility
    torch.manual_seed(model_random_seed)
    random.seed(model_random_seed)
    np.random.seed(model_random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Datasets
    TRAIN_dataset = GenomeDataset(ins1_path, ins2_path, bound_path, genome_path, demean_context_bin_width, DNA_context_bin_width, boundary_length, train_coords_dict, blacklist_path)
    VAL_dataset = GenomeDataset(ins1_path, ins2_path, bound_path, genome_path, demean_context_bin_width, DNA_context_bin_width, boundary_length, val_coords_dict, blacklist_path)
    print('Train dataset', len(TRAIN_dataset))
    print('Val dataset', len(VAL_dataset))

    # Dataloaders
    def seed_worker(worker_id):
        """Initializes random seed for each worker."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(model_random_seed)

    TRAIN_dataloader = torch.utils.data.DataLoader(
        TRAIN_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
        worker_init_fn=seed_worker, generator=g,
    )
    VAL_dataloader = torch.utils.data.DataLoader(
        VAL_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Model
    context_length = boundary_length * DNA_context_bin_width # context on one arm in bp

    if model_type == "DominoSingleOutput":
        model = domino.models.domino_models.DominoSingleOutput(
            context_length=context_length,
            boundary_length=boundary_length,
            ins_window_size=ins_window_size, 
            demean_context_bin_width=demean_context_bin_width,
            delta=delta,
            model_random_seed=model_random_seed,
            conv_layers=conv_layers,
            conv_repeat=conv_repeat,
            kernel_number=kernel_number,
            kernel_length=kernel_length,
            filter_number=filter_number,
            filter_size=filter_size,
            h_layers=h_layers,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            scheduler=scheduler,
            activation=activation
        )
    else:
        raise ValueError("Model type not supported")

    # Hooks
    es = EarlyStopping(monitor="val_corr", patience=20, mode="max")
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path, monitor="val_corr", mode="max", save_top_k=1
    )
    lr_monitor = LearningRateMonitor()
    logger = TensorBoardLogger(log_path, name=model_type)

    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        callbacks=[es, checkpoint_callback, lr_monitor],
        benchmark=False,
        profiler="simple",
        logger=logger,
        max_epochs=200,  # 200, 400
        val_check_interval=0.25,
    )

    trainer.fit(model, TRAIN_dataloader, val_dataloaders=VAL_dataloader)

    ###################################################################################################
    # Evaluation (on test set)
    ###################################################################################################

    TEST_dataset = GenomeDataset(ins1_path, ins2_path, bound_path, genome_path, demean_context_bin_width, DNA_context_bin_width, boundary_length, test_coords_dict, blacklist_path)
    TEST_dataloader = torch.utils.data.DataLoader(
        TEST_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print(len(TEST_dataset))
    orig_prediction = trainer.predict(dataloaders = TEST_dataloader)
    prediction = torch.cat(orig_prediction, dim=0).cpu().numpy()

    pearson = stats.pearsonr(prediction[:, 0], prediction[:, 1])[0]
    spearman = stats.spearmanr(prediction[:, 0], prediction[:, 1])[0]
    mse = np.mean((prediction[:, 0] - prediction[:, 1]) ** 2)
    with open(output_path + '/test_performance.txt', 'w') as file:
        file.write(f"{pearson}\n")
        file.write(f"{spearman}\n")
        file.write(f"{mse}\n")

    plt.hexbin(prediction[:, 1], prediction[:, 0], gridsize=200, cmap='viridis', bins='log', vmin=0.15)
    max_val = max(prediction[:, 0].max(), prediction[:, 1].max())
    min_val = min(prediction[:, 0].min(), prediction[:, 1].min())
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('True mean-scaled insulation score')
    plt.ylabel('Predicted mean-scaled insulation score')
    plt.title(f'Correlation performance (test set)')
    plt.text(0.03, 0.95, f'PCC: {pearson:.3f}', transform=plt.gca().transAxes,
         ha='left', va='top', fontsize=10, color='black')
    plt.text(0.03, 0.90, f'SCC: {spearman:.3f}', transform=plt.gca().transAxes,
         ha='left', va='top', fontsize=10, color='black')
    plt.savefig(output_path + '/TEST_scatter.png')
    plt.close()

    plt.hist(prediction[:, 0], bins=100, alpha=0.5, label='Prediction')
    plt.hist(prediction[:, 1], bins=100, alpha=0.5, label='True')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Distribution of mean-scaled insulation')
    plt.savefig(output_path + '/TEST_hist.png')
    plt.close()

    pos_pearson = stats.pearsonr(prediction[prediction[:, 2] == 1, 0], prediction[prediction[:, 2] == 1, 1])[0]
    pos_spearman = stats.spearmanr(prediction[prediction[:, 2] == 1, 0], prediction[prediction[:, 2] == 1, 1])[0]
    pos_mse = np.mean((prediction[prediction[:, 2] == 1, 0] - prediction[prediction[:, 2] == 1, 1]) ** 2)

    plt.hexbin(prediction[prediction[:, 2] == 1, 1], prediction[prediction[:, 2] == 1, 0], gridsize=200, cmap='viridis', bins='log', vmin=0.15)
    max_val = max(prediction[prediction[:, 2] == 1, 0].max(), prediction[prediction[:, 2] == 1, 1].max())
    min_val = min(prediction[prediction[:, 2] == 1, 0].min(), prediction[prediction[:, 2] == 1, 1].min())
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.title(f'Correlation performance on positive examples (test set)')
    plt.text(0.03, 0.95, f'PCC: {pos_pearson:.3f}', transform=plt.gca().transAxes,
         ha='left', va='top', fontsize=10, color='black')
    plt.text(0.03, 0.90, f'SCC: {pos_spearman:.3f}', transform=plt.gca().transAxes,
         ha='left', va='top', fontsize=10, color='black')
    plt.savefig(output_path + '/TEST_pos_scatter.png')
    plt.close()

    plt.hist(prediction[prediction[:, 2] == 1, 0], bins=100, alpha=0.5, label='Prediction')
    plt.hist(prediction[prediction[:, 2] == 1, 1], bins=100, alpha=0.5, label='True')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Distribution of mean-scaled insulation (positive examples)')
    plt.savefig(output_path + '/TEST_pos_hist.png')
    plt.close()

    neg_pearson = stats.pearsonr(prediction[prediction[:, 2] == 0, 0], prediction[prediction[:, 2] == 0, 1])[0]
    neg_spearman = stats.spearmanr(prediction[prediction[:, 2] == 0, 0], prediction[prediction[:, 2] == 0, 1])[0]
    neg_mse = np.mean((prediction[prediction[:, 2] == 0, 0] - prediction[prediction[:, 2] == 0, 1]) ** 2)

    plt.hexbin(prediction[prediction[:, 2] == 0, 1], prediction[prediction[:, 2] == 0, 0], gridsize=200, cmap='viridis', bins='log', vmin=0.15)
    max_val = max(prediction[prediction[:, 2] == 0, 0].max(), prediction[prediction[:, 2] == 0, 1].max())
    min_val = min(prediction[prediction[:, 2] == 0, 0].min(), prediction[prediction[:, 2] == 0, 1].min())
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.title(f'Correlation performance on negative examples (test set)')
    plt.text(0.03, 0.95, f'PCC: {neg_pearson:.3f}', transform=plt.gca().transAxes,
         ha='left', va='top', fontsize=10, color='black')
    plt.text(0.03, 0.90, f'SCC: {neg_spearman:.3f}', transform=plt.gca().transAxes,
         ha='left', va='top', fontsize=10, color='black')
    plt.savefig(output_path + '/TEST_neg_scatter.png')
    plt.close()

    plt.hist(prediction[prediction[:, 2] == 0, 0], bins=100, alpha=0.5, label='Prediction')
    plt.hist(prediction[prediction[:, 2] == 0, 1], bins=100, alpha=0.5, label='True')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Distribution of mean-scaled insulation (negative examples)')
    plt.savefig(output_path + '/TEST_neg_hist.png')
    plt.close()

    test_df = TEST_dataset.df
    pearson = stats.pearsonr(test_df['demeaned_ins_score_1'], test_df['demeaned_ins_score_2'])[0]
    spearman = stats.spearmanr(test_df['demeaned_ins_score_1'], test_df['demeaned_ins_score_2'])[0]
    mse = np.mean((test_df['demeaned_ins_score_1'] - test_df['demeaned_ins_score_2']) ** 2)

    plt.hexbin(test_df['demeaned_ins_score_1'], test_df['demeaned_ins_score_2'], gridsize=200, cmap='viridis', bins='log', vmin=0.15)
    max_val = max(test_df['demeaned_ins_score_1'].max(), test_df['demeaned_ins_score_2'].max())
    min_val = min(test_df['demeaned_ins_score_1'].min(), test_df['demeaned_ins_score_2'].min())
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Mean-scaled insulation score (rep 1)')
    plt.ylabel('Mean-scaled insulation score (rep 2)')
    plt.title(f'Replicate performance (test set)')
    plt.text(0.03, 0.95, f'PCC: {pearson:.3f}', transform=plt.gca().transAxes,
        ha='left', va='top', fontsize=10, color='black')
    plt.text(0.03, 0.90, f'SCC: {spearman:.3f}', transform=plt.gca().transAxes,
        ha='left', va='top', fontsize=10, color='black')
    plt.savefig(output_path + '/TEST_rep_scatter.png')
    plt.close()

    ###################################################################################################
    # Save dataframes with predictions
    ###################################################################################################

    # need to redefine because data loaders must have shuffle set to false
    # since positive examples have even indices and negative odd
    TRAIN_dataloader = torch.utils.data.DataLoader(
        TRAIN_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    VAL_dataloader = torch.utils.data.DataLoader(
        VAL_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    TEST_dataloader = torch.utils.data.DataLoader(
        TEST_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_prediction = trainer.predict(dataloaders=VAL_dataloader)
    train_prediction = trainer.predict(dataloaders=TRAIN_dataloader)
    test_prediction = trainer.predict(dataloaders=TEST_dataloader)

    val_prediction = torch.cat(val_prediction, dim=0).cpu().numpy()
    val_prediction = val_prediction.reshape(-1, 3)
    print(val_prediction.shape)

    train_prediction = torch.cat(train_prediction, dim=0).cpu().numpy()
    train_prediction = train_prediction.reshape(-1, 3)
    print(train_prediction.shape)

    test_prediction = torch.cat(test_prediction, dim=0).cpu().numpy()
    test_prediction = test_prediction.reshape(-1, 3)
    print(test_prediction.shape)

    val_df = VAL_dataset.df
    pearson = stats.pearsonr(val_df['demeaned_ins_score_1'], val_df['demeaned_ins_score_2'])[0]
    spearman = stats.spearmanr(val_df['demeaned_ins_score_1'], val_df['demeaned_ins_score_2'])[0]
    mse = np.mean((val_df['demeaned_ins_score_1'] - val_df['demeaned_ins_score_2']) ** 2)

    plt.hexbin(val_df['demeaned_ins_score_1'], val_df['demeaned_ins_score_2'], gridsize=200, cmap='viridis', bins='log', vmin=0.15)
    max_val = max(val_df['demeaned_ins_score_1'].max(), val_df['demeaned_ins_score_2'].max())
    min_val = min(val_df['demeaned_ins_score_1'].min(), val_df['demeaned_ins_score_2'].min())
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Mean-scaled insulation score (rep 1)')
    plt.ylabel('Mean-scaled insulation score (rep 2)')
    plt.title(f'Replicate performance (test set)')
    plt.text(0.03, 0.95, f'PCC: {pearson:.3f}', transform=plt.gca().transAxes,
        ha='left', va='top', fontsize=10, color='black')
    plt.text(0.03, 0.90, f'SCC: {spearman:.3f}', transform=plt.gca().transAxes,
        ha='left', va='top', fontsize=10, color='black')
    plt.savefig(output_path + '/VAL_rep_scatter.png')
    plt.close()

    pearson = stats.pearsonr(val_prediction[:, 0], val_prediction[:, 1])[0]
    spearman = stats.spearmanr(val_prediction[:, 0], val_prediction[:, 1])[0]
    mse = np.mean((val_prediction[:, 0] - val_prediction[:, 1]) ** 2)
    with open(output_path + '/val_performance.txt', 'w') as file:
        file.write(f"{pearson}\n")
        file.write(f"{spearman}\n")
        file.write(f"{mse}\n")

    TRAIN_df = process_pos_neg(TRAIN_dataset.df.reset_index(drop=True), train_prediction[:, 0])
    VAL_df = process_pos_neg(VAL_dataset.df.reset_index(drop=True), val_prediction[:, 0])
    TEST_df = process_pos_neg(TEST_dataset.df.reset_index(drop=True), test_prediction[:, 0])
    TRAIN_df.to_csv(f'{output_path}/TRAIN_df.csv', index=False)
    VAL_df.to_csv(f'{output_path}/VAL_df.csv', index=False)
    TEST_df.to_csv(f'{output_path}/TEST_df.csv', index=False)