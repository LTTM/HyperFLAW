import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from pathlib import Path

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
    args = u.argparser("eval")

    # Get the current date
    date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    print(f"\nUsing {str(DEVICE).upper()} device\n")

    # Load the training dataset
    _, test_transform, dts = dt.get_dataset(
        dataset=dt.TestDataset,
        color_jitter=args.color_jitter,
        gaussian_blur=args.gaussian_blur,
        random_flip=args.random_flip,
    )
    test_dts = dts(
        txt_file=args.dts_txt, root=args.dataset_path, transform=test_transform
    )
    test_loader = DataLoader(test_dts, batch_size=1, shuffle=True, num_workers=10)

    # Initialize the model
    network = u.net_picker(args.model)
    model = network(num_classes=4).to(DEVICE)

    # Load the model
    model.load_state_dict(torch.load(args.checkpoint_path))

    # Define the path to store the data
    test_path = Path("./tests")
    current_test_path = Path(
        test_path / f"{model.__class__.__name__}_{date}"
    )
    current_test_path.mkdir(parents=True, exist_ok=True)

    # Define the trainer
    trainer = Trainer(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
    )

    # Test the model
    out = trainer.test()

    # Plot the confusion matrix
    u.plot_confusion_matrix(
        out['c_matrix'],
        current_test_path / "confusion_matrix.svg",
    )

    # Save the results
    np.save(current_test_path / "predictions.npy", out['prediction'])
    np.save(current_test_path / "ground_truth.npy", out['gt'])
    np.save(current_test_path / "confusion_matrix.npy", out['c_matrix'])
    with open(current_test_path / "accuracies.txt", "w") as f:
        for k, v in zip(MAPPING_WEATHER.keys(), out["accuracy"]):
            f.write(f"  - {k}: {round(v * 100, 2)}%\n")

    # Print the results
    print("Overall accuracy:")
    for k, v in zip(MAPPING_WEATHER.keys(), out["accuracy"]):
        print(f"  - {k}: {round(v * 100, 2)}%")
