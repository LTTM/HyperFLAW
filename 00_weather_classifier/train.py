import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from modules import dataset as dt
from modules import models as m
from modules import utils as u
from modules import Trainer
from modules import MAPPING_WEATHER


# Define the device to use based on what available: CPU, GPU or METAL
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Define the random seed
SEED = 41
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


if __name__ == "__main__":
    # Parse the arguments
    args = u.argparser("train")

    # Get the current date
    date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    print(f"\nUsing {str(DEVICE).upper()} device\n")

    # Load the training dataset
    transform, test_transform, dts = dt.get_dataset(
        dataset=dt.SynSELMA,
        color_jitter=args.color_jitter,
        gaussian_blur=args.gaussian_blur,
        random_flip=args.random_flip,
    )
    train_dts = dts(root=args.dataset_path, transform=transform, split="train")
    test_dts = dts(root=args.dataset_path, test_transform=test_transform, split="test")

    train_loader = DataLoader(train_dts, batch_size=args.batch_size, shuffle=True, num_workers=10)
    test_loader = DataLoader(test_dts, batch_size=1, shuffle=True, num_workers=10)

    # Initialize the model and optimizer
    network = u.net_picker(args.model)
    model = network(num_classes=4).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(reduction="mean").to(DEVICE)

    # Define the path to store the data
    checkpoin_path = Path("./checkpoints")
    current_checkpoint_path = Path(
        checkpoin_path / f"{model.__class__.__name__}_{date}"
    )
    current_checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Define the trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        ckpt_path=current_checkpoint_path,
    )

    # Train the model
    out = trainer.train(args.num_epochs)

    # Save the model and raining statistics
    for k, v in out.items():
        if k not in ["best_accuracies", "best_epoch"]:
            np.save(current_checkpoint_path / f"{k}.npy", v)

    # Plot the best accuracies class-wise
    u.plot_losses(
        [out["train_losses"], out["eval_losses"]],
        current_checkpoint_path / "losses.svg",
    )
    u.plot_accuracies(
        [out["train_accuracies"], out["eval_accuracies"]],
        current_checkpoint_path / "accuracies.svg",
    )
    u.plot_confusion_matrix(
        out["best_cf_matrix"], current_checkpoint_path / "best_confusion_matrix.svg"
    )

    # Print the overall accuracy for each weather condition
    print(f'\nBest epoch: {out["best_epoch"]}')
    print("Overall accuracies:")
    for k, v in zip(MAPPING_WEATHER.keys(), out["best_accuracies"]):
        print(f"  - {k}: {round(v * 100, 2)}%")

    # Save the overall accuracies to file
    with open(current_checkpoint_path / "best_accuracies.txt", "w") as f:
        f.write(f'Best epoch: {out["best_epoch"]}\n\n')
        for k, v in zip(MAPPING_WEATHER.keys(), out["best_accuracies"]):
            f.write(f"  - {k}: {round(v * 100, 2)}%\n")
